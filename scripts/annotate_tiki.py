"""
Annotate aspect_term + aspect_term_span for Tiki LLM-labeled dataset.

Input:  train_clean.json + test_clean.json from the Tiki midterm dataset
Output: data/interim/tiki_aspect_term_draft.jsonl  (auto-annotated)
        data/interim/tiki_validation_sample.csv     (300 records for human review)

Tiki aspect taxonomy mapping:
  as_content  -> product_quality
  as_physical -> appearance
  as_price    -> price
  as_packaging-> packaging
  as_delivery -> delivery
  as_service  -> customer_service

Sentiment: 0.0=negative, 1.0=neutral (dropped), 2.0=positive

Usage:
    python scripts/annotate_tiki.py --tiki-dir PATH_TO_DM_VERSION3
    python scripts/annotate_tiki.py --tiki-dir PATH --dry-run
    python scripts/annotate_tiki.py --tiki-dir PATH --batch-size 10 --delay 4.5
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

TIKI_FILES = ["data/train_clean.json", "data/test_clean.json"]
INTERIM_DIR = Path("data/interim")
DRAFT_PATH = INTERIM_DIR / "tiki_aspect_term_draft.jsonl"
VALIDATION_CSV = INTERIM_DIR / "tiki_validation_sample.csv"
VALIDATION_N = 300

ASPECT_MAP = {
    "as_content":   "product_quality",
    "as_physical":  "appearance",
    "as_price":     "price",
    "as_packaging": "packaging",
    "as_delivery":  "delivery",
    "as_service":   "customer_service",
}
SENTIMENT_MAP = {0.0: "negative", 2.0: "positive"}  # 1.0 neutral -> dropped

ASPECT_HINTS = {
    "product_quality":  "chất lượng, chất liệu, vải, da, nhựa, sản phẩm, hàng, nội dung, sách",
    "appearance":       "màu sắc, kiểu dáng, hình thức, thiết kế, mẫu mã, bìa, hình ảnh",
    "price":            "giá, tiền, chi phí, giá cả, giá tiền",
    "packaging":        "đóng gói, bao bì, hộp, túi, bọc, bong bóng, tem",
    "delivery":         "giao hàng, ship, vận chuyển, giao, shipper, thời gian giao",
    "customer_service": "nhân viên, tư vấn, hỗ trợ, shop, người bán, dịch vụ, Tiki",
}

ASPECT_KEYWORDS = {
    "product_quality":  ["sản phẩm", "hàng", "chất lượng", "chất liệu", "nội dung", "sách"],
    "appearance":       ["bìa", "hình thức", "màu sắc", "kiểu dáng", "thiết kế"],
    "price":            ["giá", "tiền", "chi phí", "giá cả"],
    "packaging":        ["đóng gói", "bao bì", "hộp", "túi", "bọc"],
    "delivery":         ["giao hàng", "giao", "ship", "vận chuyển", "shipper"],
    "customer_service": ["nhân viên", "Tiki", "shop", "hỗ trợ", "dịch vụ"],
}


def call_gemini(prompt: str, api_key: str, model: str = "gemini-2.0-flash") -> str:
    import urllib.request
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048},
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def find_span(review: str, term: str) -> tuple[int, int] | None:
    idx = review.find(term)
    return (idx, idx + len(term)) if idx != -1 else None


def heuristic_fallback(review: str, aspect_category: str) -> tuple[str, int, int]:
    for kw in ASPECT_KEYWORDS.get(aspect_category, []):
        sp = find_span(review, kw)
        if sp:
            return kw, sp[0], sp[1]
    dummy = aspect_category
    return dummy, 0, min(len(dummy), len(review))


def build_batch_prompt(records: list[dict]) -> str:
    lines = []
    for i, r in enumerate(records):
        hint = ASPECT_HINTS.get(r["aspect_category"], "")
        lines.append(
            f'[{i}] review: "{r["review"]}" | aspect: {r["aspect_category"]} | hints: {hint}'
        )
    return (
        "Bạn là công cụ annotation tiếng Việt. Nhiệm vụ: tìm cụm từ NGẮN NHẤT "
        "trong review chỉ ĐỐI TƯỢNG của aspect (không phải tính từ mô tả).\n\n"
        "Ví dụ:\n"
        '- "giao hàng hơi chậm" + delivery -> aspect_term = "giao hàng"\n'
        '- "chất vải mịn" + product_quality -> aspect_term = "chất vải"\n'
        '- "sách in đẹp" + appearance -> aspect_term = "sách"\n'
        '- "giá hợp lý" + price -> aspect_term = "giá"\n\n'
        "Nếu không có noun rõ ràng, lấy verb/keyword ngắn nhất chỉ đối tượng đó.\n"
        "BẮT BUỘC: review[start:end] == aspect_term chính xác.\n\n"
        "Entries:\n" + "\n".join(lines) + "\n\n"
        'Trả về JSON array KHÔNG có giải thích:\n'
        '[{"idx": 0, "aspect_term": "...", "start": <int>, "end": <int>}, ...]'
    )


def process_batch_gemini(batch: list[dict], api_key: str) -> dict[int, dict]:
    """Returns {local_idx: {aspect_term, start, end}} from Gemini. Raises HTTPError on 429."""
    import urllib.error
    raw = call_gemini(build_batch_prompt(batch), api_key)
    match = re.search(r'\[[\s\S]*\]', raw)
    if not match:
        return {}
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return {}
    out: dict[int, dict] = {}
    for item in items:
        idx = int(item.get("idx", -1))
        if 0 <= idx < len(batch):
            out[idx] = item
    return out


def resolve_batch(batch: list[dict], gemini_results: dict[int, dict]) -> list[dict]:
    results = []
    for local_idx, rec in enumerate(batch):
        source = "heuristic"
        term, start, end = heuristic_fallback(rec["review"], rec["aspect_category"])

        if local_idx in gemini_results:
            g = gemini_results[local_idx]
            g_term = g.get("aspect_term", "")
            g_start = int(g.get("start", -1))
            g_end = int(g.get("end", -1))
            review = rec["review"]
            if 0 <= g_start < g_end <= len(review) and review[g_start:g_end] == g_term:
                term, start, end = g_term, g_start, g_end
                source = "llm_batch"
            else:
                sp = find_span(review, g_term)
                if sp:
                    term, start, end = g_term, sp[0], sp[1]
                    source = "llm_batch"

        valid = (0 <= start < end <= len(rec["review"]) and rec["review"][start:end] == term)
        results.append({
            "id": rec["id"],
            "ann_idx": rec["ann_idx"],
            "review": rec["review"],
            "aspect_term": term,
            "aspect_term_span": [start, end],
            "aspect_category": rec["aspect_category"],
            "sentiment": rec["sentiment"],
            "source": source,
            "valid": valid,
        })
    return results


def load_tiki_records(tiki_dir: Path) -> list[dict]:
    records: list[dict] = []
    for fname in TIKI_FILES:
        fpath = tiki_dir / fname
        if not fpath.exists():
            print(f"WARNING: {fpath} not found — skipping")
            continue
        data = json.loads(fpath.read_text(encoding="utf-8"))
        for r in data:
            review = r.get("content", "").strip()
            if not review:
                continue
            rid = str(r.get("review_id", ""))
            ann_idx = 0
            for tiki_col, cat in ASPECT_MAP.items():
                val = r.get(tiki_col)
                if val is None:
                    continue
                sentiment = SENTIMENT_MAP.get(float(val))
                if sentiment is None:
                    continue  # neutral
                records.append({
                    "id": f"tiki-{rid}",
                    "ann_idx": ann_idx,
                    "review": review,
                    "aspect_category": cat,
                    "sentiment": sentiment,
                })
                ann_idx += 1
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tiki-dir",
        default=r"D:\Study\College\ThirdYear\SecondSemester\DataMining\Midterm\dm_version3",
    )
    ap.add_argument("--api-key", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--delay", type=float, default=4.5, help="Seconds between batches (15 RPM ~ 4s)")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N records (0=all)")
    args = ap.parse_args()

    api_key = args.api_key
    if not api_key and not args.dry_run:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
        if not api_key:
            raise SystemExit("ERROR: GEMINI_API_KEY not found")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    tiki_dir = Path(args.tiki_dir)

    all_records = load_tiki_records(tiki_dir)
    print(f"Loaded {len(all_records)} non-neutral Tiki annotations")

    if args.limit:
        all_records = all_records[: args.limit]
        print(f"Limited to first {args.limit} records")

    # Resume support
    done_keys: set[str] = set()
    if DRAFT_PATH.exists():
        for line in DRAFT_PATH.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                done_keys.add(obj["id"] + "_" + str(obj["ann_idx"]))
            except Exception:
                pass
        if done_keys:
            print(f"Resuming: {len(done_keys)} already done")

    pending = [r for r in all_records if r["id"] + "_" + str(r["ann_idx"]) not in done_keys]
    n_batches = (len(pending) + args.batch_size - 1) // args.batch_size
    print(f"Pending: {len(pending)} annotations ({n_batches} batches of {args.batch_size})")

    total_written = 0
    quota_hit = False

    with DRAFT_PATH.open("a", encoding="utf-8") as f:
        for batch_num, batch_start in enumerate(range(0, len(pending), args.batch_size), 1):
            batch = pending[batch_start: batch_start + args.batch_size]

            if args.dry_run or quota_hit:
                results = resolve_batch(batch, {})
            else:
                import urllib.error
                gemini_results: dict[int, dict] = {}
                for attempt in range(4):
                    try:
                        gemini_results = process_batch_gemini(batch, api_key)
                        break
                    except urllib.error.HTTPError as e:
                        if e.code == 429:
                            if attempt < 3:
                                wait = 60 * (attempt + 1)
                                print(f"\n  [429] wait {wait}s (attempt {attempt+1}/4)...")
                                time.sleep(wait)
                            else:
                                print("\n  [QUOTA] Daily limit — switching to heuristic for all remaining")
                                quota_hit = True
                        else:
                            print(f"\n  [HTTP {e.code}] — heuristic for this batch")
                            break
                results = resolve_batch(batch, gemini_results)

            for out in results:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
            total_written += len(results)

            if batch_num % 50 == 0 or batch_start + args.batch_size >= len(pending):
                done_total = len(done_keys) + total_written
                src = results[0]["source"] if results else "?"
                print(f"  [{done_total}/{len(all_records)}] batch {batch_num}/{n_batches} source={src}")

            if not args.dry_run and not quota_hit and batch_start + args.batch_size < len(pending):
                time.sleep(args.delay)

    print(f"\nDraft -> {DRAFT_PATH} (+{total_written} records)")

    all_draft = []
    for line in DRAFT_PATH.read_text(encoding="utf-8").splitlines():
        try:
            all_draft.append(json.loads(line))
        except Exception:
            pass

    sample = random.sample(all_draft, min(VALIDATION_N, len(all_draft)))
    with VALIDATION_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "ann_idx", "review", "aspect_category", "sentiment",
            "aspect_term_llm", "span_start", "span_end", "span_text_check",
            "source", "your_correction", "correct_start", "correct_end", "notes",
        ])
        writer.writeheader()
        for row in sample:
            s, e = row["aspect_term_span"]
            writer.writerow({
                "id": row["id"],
                "ann_idx": row["ann_idx"],
                "review": row["review"],
                "aspect_category": row["aspect_category"],
                "sentiment": row["sentiment"],
                "aspect_term_llm": row["aspect_term"],
                "span_start": s,
                "span_end": e,
                "span_text_check": row["review"][s:e],
                "source": row.get("source", "llm"),
                "your_correction": "",
                "correct_start": "",
                "correct_end": "",
                "notes": "",
            })

    print(f"Validation CSV -> {VALIDATION_CSV} ({len(sample)} records)")
    print("\nNext steps:")
    print("  1. Human review data/interim/tiki_validation_sample.csv")
    print("  2. python scripts/merge_corrections.py --validation data/interim/tiki_validation_sample.csv")
    print("  3. python scripts/build_training_set.py")


if __name__ == "__main__":
    main()

"""
Patch the 31 missing annotations in gold_aspect_term_draft.jsonl.
Batches all missing records into a single Gemini API call (31 records << 1M token limit).
Falls back to cause_text-based heuristic for any that still fail.

Usage:
    python scripts/patch_missing_annotations.py
    python scripts/patch_missing_annotations.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

GOLD_SPLITS = [
    "data/gold/train.json",
    "data/gold/val.json",
    "data/gold/test.json",
]
INTERIM_DIR = Path("data/interim")
DRAFT_PATH = INTERIM_DIR / "gold_aspect_term_draft.jsonl"
VALIDATION_CSV = INTERIM_DIR / "validation_sample.csv"
VALIDATION_N = 300

ASPECT_HINTS = {
    "delivery":         "giao hàng, ship, vận chuyển, giao, shipper, thời gian giao",
    "packaging":        "đóng gói, bao bì, hộp, túi, bọc, bong bóng",
    "product_quality":  "chất lượng, chất liệu, vải, da, nhựa, sản phẩm, hàng",
    "price":            "giá, tiền, chi phí, giá cả, giá tiền",
    "customer_service": "nhân viên, tư vấn, hỗ trợ, shop, người bán, dịch vụ",
    "usability":        "sử dụng, dùng, trải nghiệm, tính năng, hoạt động",
    "appearance":       "màu sắc, kiểu dáng, hình thức, thiết kế, mẫu mã, ngoại hình",
}

ASPECT_KEYWORDS = {
    "delivery":         ["giao hàng", "giao", "ship", "vận chuyển", "shipper"],
    "packaging":        ["đóng gói", "bao bì", "hộp", "túi", "bọc"],
    "product_quality":  ["chất lượng", "sản phẩm", "hàng", "chất liệu"],
    "price":            ["giá", "tiền", "chi phí", "giá cả"],
    "customer_service": ["nhân viên", "phục vụ", "tư vấn", "hỗ trợ", "dịch vụ"],
    "usability":        ["sử dụng", "dùng", "tính năng", "wifi", "pin"],
    "appearance":       ["không gian", "vị trí", "thiết kế", "quán", "phòng"],
}


def call_gemini(prompt: str, api_key: str, model: str = "gemini-2.0-flash") -> str:
    import urllib.request
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
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


def heuristic_fallback(review: str, aspect_category: str, cause_text: str) -> tuple[str, int, int]:
    """Extract aspect_term from cause_text using keyword matching as last resort."""
    # Try aspect-specific keywords first
    for kw in ASPECT_KEYWORDS.get(aspect_category, []):
        sp = find_span(review, kw)
        if sp:
            return kw, sp[0], sp[1]

    # Try progressively shorter prefixes of cause_text
    words = cause_text.split()
    for n in range(min(4, len(words)), 0, -1):
        phrase = " ".join(words[:n])
        sp = find_span(review, phrase)
        if sp:
            return phrase, sp[0], sp[1]

    # Last resort: first word of cause_text that appears in review
    for w in words:
        if len(w) >= 2:
            sp = find_span(review, w)
            if sp:
                return w, sp[0], sp[1]

    # Absolute fallback: position 0, use first word of cause_text
    term = words[0] if words else cause_text[:4]
    return term, 0, len(term)


def build_batch_prompt(records: list[dict]) -> str:
    lines = []
    for i, r in enumerate(records):
        lines.append(
            f'[{i}] review: "{r["review"]}" | aspect: {r["aspect"]} '
            f'| cause_text: "{r["cause_text"]}" | hints: {ASPECT_HINTS.get(r["aspect"], "")}'
        )

    return (
        "Bạn là công cụ annotation tiếng Việt. Nhiệm vụ: tìm cụm từ NGẮN NHẤT "
        "trong review chỉ ĐỐI TƯỢNG của aspect (không phải tính từ mô tả).\n\n"
        "Ví dụ:\n"
        '- "giao hàng hơi chậm" + delivery → aspect_term = "giao hàng"\n'
        '- "chất vải mịn" + product_quality → aspect_term = "chất vải"\n'
        '- "nhân viên nhiệt tình" + customer_service → aspect_term = "nhân viên"\n\n'
        "Với MỖI entry bên dưới, trả về JSON object. "
        "BẮT BUỘC: review[start:end] == aspect_term chính xác.\n\n"
        "Entries:\n" + "\n".join(lines) + "\n\n"
        'Trả về JSON array KHÔNG có giải thích:\n'
        '[{"idx": 0, "aspect_term": "...", "start": <int>, "end": <int>}, ...]'
    )


def load_missing(done_keys: set[str]) -> list[dict]:
    missing = []
    for split_path in GOLD_SPLITS:
        p = Path(split_path)
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for r in data:
            non_neutral = [a for a in r.get("annotations", []) if a["sentiment"] != "neutral"]
            for i, ann in enumerate(non_neutral):
                key = r["id"] + "_" + str(i)
                if key not in done_keys:
                    missing.append({
                        "id": r["id"],
                        "review": r["review"],
                        "ann_idx": i,
                        "aspect": ann["aspect"],
                        "sentiment": ann["sentiment"],
                        "cause_text": ann.get("cause_text", ""),
                    })
    return missing


def process_batch_with_gemini(batch: list[dict], api_key: str) -> dict[int, dict]:
    """Call Gemini with a batch; returns {local_idx: {aspect_term, start, end}} or {}."""
    import urllib.error
    prompt = build_batch_prompt(batch)
    try:
        raw = call_gemini(prompt, api_key)
        # Extract JSON array
        match = re.search(r'\[[\s\S]*\]', raw)
        if not match:
            print(f"  [WARN] No JSON array in Gemini response")
            return {}
        results = json.loads(match.group())
        out = {}
        for item in results:
            idx = int(item["idx"])
            if idx >= len(batch):
                continue
            r = batch[idx]
            term = item.get("aspect_term", "")
            start = int(item.get("start", -1))
            end = int(item.get("end", -1))
            # Validate/repair span
            if not (0 <= start < end <= len(r["review"]) and r["review"][start:end] == term):
                sp = find_span(r["review"], term)
                if sp:
                    start, end = sp
                else:
                    # Try fallback
                    term, start, end = heuristic_fallback(r["review"], r["aspect"], r["cause_text"])
            out[idx] = {"aspect_term": term, "start": start, "end": end}
        return out
    except urllib.error.HTTPError as e:
        print(f"  [HTTP {e.code}] Gemini error — will use heuristic fallback for entire batch")
        return {}
    except Exception as e:
        print(f"  [ERROR] {e} — will use heuristic fallback")
        return {}


def generate_validation_csv() -> None:
    all_draft = []
    for line in DRAFT_PATH.read_text(encoding="utf-8").splitlines():
        try:
            all_draft.append(json.loads(line))
        except Exception:
            pass

    sample = random.sample(all_draft, min(VALIDATION_N, len(all_draft)))
    VALIDATION_CSV.parent.mkdir(parents=True, exist_ok=True)
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
                "aspect_category": row.get("aspect_category", row.get("aspect", "")),
                "sentiment": row["sentiment"],
                "aspect_term_llm": row["aspect_term"],
                "span_start": s,
                "span_end": e,
                "span_text_check": row["review"][s:e],
                "source": row.get("source", row.get("source_batch", "llm")),
                "your_correction": "",
                "correct_start": "",
                "correct_end": "",
                "notes": "",
            })
    print(f"Validation CSV → {VALIDATION_CSV} ({len(sample)} records)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch-size", type=int, default=15, help="Annotations per Gemini call")
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

    # Load done keys
    done_keys: set[str] = set()
    if DRAFT_PATH.exists():
        for line in DRAFT_PATH.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                done_keys.add(obj["id"] + "_" + str(obj["ann_idx"]))
            except Exception:
                pass
    print(f"Already done: {len(done_keys)} annotations")

    missing = load_missing(done_keys)
    print(f"Missing: {len(missing)} annotations — processing in batches of {args.batch_size}")

    if not missing:
        print("Nothing to patch — regenerating validation CSV only")
        generate_validation_csv()
        return

    new_records: list[dict] = []

    with DRAFT_PATH.open("a", encoding="utf-8") as f:
        for batch_start in range(0, len(missing), args.batch_size):
            batch = missing[batch_start: batch_start + args.batch_size]
            print(f"  Batch [{batch_start}:{batch_start + len(batch)}]", end=" ")

            if args.dry_run:
                gemini_results = {}
            else:
                gemini_results = process_batch_with_gemini(batch, api_key)

            print(f"→ Gemini answered {len(gemini_results)}/{len(batch)}")

            for local_idx, rec in enumerate(batch):
                if local_idx in gemini_results and not args.dry_run:
                    g = gemini_results[local_idx]
                    term, start, end = g["aspect_term"], g["start"], g["end"]
                    source = "llm_batch"
                else:
                    term, start, end = heuristic_fallback(
                        rec["review"], rec["aspect"], rec["cause_text"]
                    )
                    source = "heuristic"

                valid = (0 <= start < end <= len(rec["review"]) and
                         rec["review"][start:end] == term)
                out = {
                    "id": rec["id"],
                    "ann_idx": rec["ann_idx"],
                    "review": rec["review"],
                    "aspect_term": term,
                    "aspect_term_span": [start, end],
                    "aspect_category": rec["aspect"],
                    "sentiment": rec["sentiment"],
                    "source": source,
                    "valid": valid,
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                new_records.append(out)

    print(f"\nAdded {len(new_records)} patched records to {DRAFT_PATH}")

    # Verify final count
    total = sum(1 for _ in DRAFT_PATH.read_text(encoding="utf-8").splitlines() if _.strip())
    print(f"Total draft records: {total}")

    generate_validation_csv()
    print("\nNext steps:")
    print("  1. Open data/interim/validation_sample.csv — human review 300 records")
    print("  2. Run: python scripts/merge_corrections.py")


if __name__ == "__main__":
    main()

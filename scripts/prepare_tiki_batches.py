"""
Convert Tiki midterm dataset → CausaSent schema and split into annotation
batches.

Source: D:/Study/College/ThirdYear/SecondSemester/DataMining/Midterm/dm_version3/data/
        train_clean.json + test_clean.json
        (each record has `content`, `sentiment_llm`, and 6 aspect columns
         `as_content / as_physical / as_price / as_packaging / as_delivery /
         as_service` each null OR 0/1/2)

Mapping (per DECISIONS.md §2):
    as_content   → product_quality
    as_physical  → appearance
    as_price     → price
    as_packaging → packaging
    as_delivery  → delivery
    as_service   → customer_service
    (Tiki has no `usability` column — that aspect will only appear in gold)

Sentiment:  0 → negative, 1 → positive, 2 → neutral (DROP per binary policy)

Output:
    data/interim/tiki_batches/batch_NNN.json    — same schema as gold batches
    data/interim/tiki_batches/MANIFEST.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TIKI_DIR = Path("D:/Study/College/ThirdYear/SecondSemester/DataMining/Midterm/dm_version3/data")
TIKI_FILES = ["train_clean.json", "test_clean.json"]

BATCH_DIR = Path("data/interim/tiki_batches")

ASPECT_MAP = {
    "as_content":   "product_quality",
    "as_physical":  "appearance",
    "as_price":     "price",
    "as_packaging": "packaging",
    "as_delivery":  "delivery",
    "as_service":   "customer_service",
}

SENTIMENT_MAP = {0: "negative", 1: "positive"}  # 2 (neutral) intentionally absent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap total annotations (0 = no cap)")
    args = ap.parse_args()

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    flat: list[dict] = []
    seen_review_ids: set[str] = set()

    for fname in TIKI_FILES:
        path = TIKI_DIR / fname
        if not path.exists():
            print(f"WARNING: {path} missing")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data:
            content = row.get("content") or row.get("content_raw")
            if not content or not content.strip():
                continue
            rid = str(row.get("review_id", ""))
            if rid in seen_review_ids:
                continue
            seen_review_ids.add(rid)

            ann_idx = 0
            for col, aspect_cat in ASPECT_MAP.items():
                val = row.get(col)
                if val is None:
                    continue
                try:
                    sent_int = int(val)
                except (TypeError, ValueError):
                    continue
                if sent_int not in SENTIMENT_MAP:
                    continue  # drops neutral
                flat.append({
                    "key": f"tiki-{rid}_{ann_idx}",
                    "review": content,
                    "aspect_category": aspect_cat,
                    "sentiment": SENTIMENT_MAP[sent_int],
                    # No cause_text — Tiki midterm only has category-level labels
                    "cause_text": "",
                })
                ann_idx += 1

    print(f"Flattened {len(flat)} Tiki annotations from {len(seen_review_ids)} reviews")

    if args.limit:
        flat = flat[: args.limit]
        print(f"Limited to first {args.limit}")

    n = args.batch_size
    batches = [flat[i:i + n] for i in range(0, len(flat), n)]

    manifest = []
    for i, batch in enumerate(batches, start=1):
        fname = f"batch_{i:03d}.json"
        (BATCH_DIR / fname).write_text(
            json.dumps(batch, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest.append({"batch": fname, "count": len(batch)})

    (BATCH_DIR / "MANIFEST.json").write_text(
        json.dumps({
            "total_records": len(flat),
            "batch_size": n,
            "num_batches": len(batches),
            "batches": manifest,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {len(batches)} batches to {BATCH_DIR}/")


if __name__ == "__main__":
    main()

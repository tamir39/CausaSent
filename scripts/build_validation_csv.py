"""
Build validation_sample.csv for human review.

Input:  data/interim/gold_aspect_term_draft.jsonl
Output: data/interim/validation_sample.csv

Sample 300 records stratified by aspect_category × sentiment so each cell gets
proportional representation. Human reviewer edits `your_correction` /
`correct_start` / `correct_end` / `notes` columns; merge_corrections.py then
applies edits back to the draft.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

DRAFT_PATH = Path("data/interim/gold_aspect_term_draft.jsonl")
VALIDATION_CSV = Path("data/interim/validation_sample.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="Target sample size")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    if not DRAFT_PATH.exists():
        raise SystemExit(f"ERROR: {DRAFT_PATH} not found. Run merge_annotation_batches.py first.")

    records = []
    for line in DRAFT_PATH.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

    print(f"Loaded {len(records)} draft records")

    # Stratified sample by (aspect_category, sentiment)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        buckets[(r["aspect_category"], r["sentiment"])].append(r)

    random.seed(args.seed)
    target = args.n
    total = len(records)

    sample: list[dict] = []
    for cell, rows in buckets.items():
        proportion = len(rows) / total
        take = max(1, round(target * proportion))
        take = min(take, len(rows))
        sample.extend(random.sample(rows, take))

    # Trim/pad to exactly `target`
    if len(sample) > target:
        random.shuffle(sample)
        sample = sample[:target]
    elif len(sample) < target:
        remaining = [r for r in records if r not in sample]
        sample.extend(random.sample(remaining, min(target - len(sample), len(remaining))))

    random.shuffle(sample)

    VALIDATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    with VALIDATION_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "ann_idx", "review",
            "aspect_category", "sentiment",
            "aspect_term_llm", "span_start", "span_end", "span_text_check",
            "your_correction", "correct_start", "correct_end", "notes",
        ])
        writer.writeheader()
        for r in sample:
            s, e = r["aspect_term_span"]
            writer.writerow({
                "id": r["id"],
                "ann_idx": r["ann_idx"],
                "review": r["review"],
                "aspect_category": r["aspect_category"],
                "sentiment": r["sentiment"],
                "aspect_term_llm": r["aspect_term"],
                "span_start": s,
                "span_end": e,
                "span_text_check": r["review"][s:e],
                "your_correction": "",
                "correct_start": "",
                "correct_end": "",
                "notes": "",
            })

    print(f"Wrote {len(sample)} validation records → {VALIDATION_CSV}")
    print("\nNext steps:")
    print("  1. Open validation_sample.csv in Excel/LibreOffice")
    print("  2. Check 'span_text_check' for each row")
    print("  3. If wrong: fill 'your_correction' (text), 'correct_start', 'correct_end'")
    print("  4. Run: python scripts/merge_corrections.py")


if __name__ == "__main__":
    main()

"""
Merge human corrections from validation_sample.csv back into gold_aspect_term_draft.jsonl,
then produce the final gold_aspect_term.jsonl ready for model training.

Human reviewers fill in:
  your_correction  - corrected aspect_term text (leave blank if LLM was correct)
  correct_start    - corrected span start (leave blank if LLM was correct)
  correct_end      - corrected span end (leave blank if LLM was correct)

Usage:
    python scripts/merge_corrections.py
    python scripts/merge_corrections.py --validation data/interim/validation_sample.csv
    python scripts/merge_corrections.py --stats-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DRAFT_PATH = Path("data/interim/gold_aspect_term_draft.jsonl")
VALIDATION_CSV = Path("data/interim/validation_sample.csv")
FINAL_PATH = Path("data/interim/gold_aspect_term.jsonl")


def load_corrections(csv_path: Path) -> dict[str, dict]:
    """Returns {id_annidx_key: {aspect_term, start, end}} for corrected rows only."""
    import csv
    corrections: dict[str, dict] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("your_correction", "").strip():
                continue
            key = row["id"] + "_" + row["ann_idx"]
            term = row["your_correction"].strip()
            try:
                start = int(row["correct_start"]) if row["correct_start"].strip() else None
                end = int(row["correct_end"]) if row["correct_end"].strip() else None
            except ValueError:
                start = end = None

            if start is None or end is None:
                start = end = -1

            corrections[key] = {"aspect_term": term, "start": start, "end": end}
    return corrections


def find_span(review: str, term: str) -> tuple[int, int] | None:
    idx = review.find(term)
    return (idx, idx + len(term)) if idx != -1 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", default=str(VALIDATION_CSV))
    ap.add_argument("--stats-only", action="store_true", help="Print stats without writing final file")
    args = ap.parse_args()

    csv_path = Path(args.validation)
    if not csv_path.exists():
        raise SystemExit(f"ERROR: Validation CSV not found: {csv_path}")
    if not DRAFT_PATH.exists():
        raise SystemExit(f"ERROR: Draft not found: {DRAFT_PATH}")

    corrections = load_corrections(csv_path)
    print(f"Human corrections loaded: {len(corrections)} records corrected")

    draft: dict[str, dict] = {}
    for line in DRAFT_PATH.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
            key = obj["id"] + "_" + str(obj["ann_idx"])
            draft[key] = obj
        except Exception:
            pass
    print(f"Draft records: {len(draft)}")

    applied = 0
    span_resolved = 0
    span_failed = 0
    for key, corr in corrections.items():
        if key not in draft:
            print(f"  [WARN] Correction key not in draft: {key}")
            continue
        rec = draft[key]
        term = corr["aspect_term"]
        start, end = corr["start"], corr["end"]

        if start == -1 or end == -1:
            sp = find_span(rec["review"], term)
            if sp:
                start, end = sp
                span_resolved += 1
            else:
                print(f"  [WARN] Cannot locate '{term}' in review {key} — skipping correction")
                span_failed += 1
                continue

        rec["aspect_term"] = term
        rec["aspect_term_span"] = [start, end]
        rec["human_corrected"] = True
        rec["valid"] = rec["review"][start:end] == term
        applied += 1

    print(f"Corrections applied: {applied} (span auto-resolved: {span_resolved}, failed: {span_failed})")

    if args.stats_only:
        llm_count = sum(1 for r in draft.values() if r.get("source_batch") or r.get("source") in (None, "llm_batch", "llm"))
        heuristic_count = sum(1 for r in draft.values() if r.get("source") == "heuristic")
        corrected_count = sum(1 for r in draft.values() if r.get("human_corrected"))
        valid_count = sum(1 for r in draft.values() if r.get("valid", True))
        print(f"\n--- Stats ---")
        print(f"Total:           {len(draft)}")
        print(f"LLM-annotated:   {llm_count}")
        print(f"Heuristic:       {heuristic_count}")
        print(f"Human-corrected: {corrected_count}")
        print(f"Valid spans:     {valid_count}")
        print(f"Invalid spans:   {len(draft) - valid_count}")
        return

    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = list(draft.values())
    with FINAL_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from collections import Counter
    aspect_counts: Counter = Counter()
    sent_counts: Counter = Counter()
    for r in records:
        cat = r.get("aspect_category", r.get("aspect", "unknown"))
        aspect_counts[cat] += 1
        sent_counts[r.get("sentiment", "?")] += 1

    print(f"\nFinal file: {FINAL_PATH} ({len(records)} records)")
    print("\nDistribution by aspect_category:")
    for cat, n in sorted(aspect_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<20} {n}")
    print("\nDistribution by sentiment:")
    for sent, n in sorted(sent_counts.items(), key=lambda x: -x[1]):
        print(f"  {sent:<12} {n}")
    print("\nNext steps:")
    print("  1. python scripts/build_training_set.py  # merge Tiki + gold -> dataset splits")
    print("  2. Upload to Kaggle for model training")


if __name__ == "__main__":
    main()

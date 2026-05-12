"""
Merge subagent validation judgments back into validation_sample.csv and emit an
agreement report for the paper.

Reads:  data/interim/validation_batches/output/val_batch_NNN_judged.jsonl
Writes: data/interim/validation_sample_judged.csv  (CSV with filled corrections)
        data/interim/agreement_report.json         (agreement stats per aspect)
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

VAL_CSV = Path("data/interim/validation_sample.csv")
OUT_DIR = Path("data/interim/validation_batches/output")
JUDGED_CSV = Path("data/interim/validation_sample_judged.csv")
REPORT_PATH = Path("data/interim/agreement_report.json")


def main() -> None:
    # Load all judgments keyed by (id, ann_idx)
    judgments: dict[tuple[str, int], dict] = {}
    for f in sorted(OUT_DIR.glob("val_batch_*_judged.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            key = (j["id"], int(j["ann_idx"]))
            judgments[key] = j

    # Load original validation CSV
    with VAL_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())

    n_total = 0
    n_agreed = 0
    n_corrected = 0
    n_missing_judgment = 0

    per_aspect: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "agreed": 0})
    correction_patterns: Counter = Counter()
    notes_by_aspect: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        key = (row["id"], int(row["ann_idx"]))
        n_total += 1
        asp = row["aspect_category"]
        per_aspect[asp]["total"] += 1

        if key not in judgments:
            n_missing_judgment += 1
            continue

        j = judgments[key]
        if j.get("agree", False):
            n_agreed += 1
            per_aspect[asp]["agreed"] += 1
        else:
            n_corrected += 1
            row["your_correction"] = j.get("correct_term") or ""
            cs = j.get("correct_start")
            ce = j.get("correct_end")
            row["correct_start"] = "" if cs is None else str(cs)
            row["correct_end"] = "" if ce is None else str(ce)
            row["notes"] = j.get("notes", "")

            llm_term = row["aspect_term_llm"]
            corrected = j.get("correct_term") or ""
            if llm_term and corrected:
                correction_patterns[f"{llm_term} → {corrected}"] += 1
            notes_by_aspect[asp].append(f"{row['id']}_{row['ann_idx']}: {j.get('notes', '')}")

    # Write judged CSV
    JUDGED_CSV.parent.mkdir(parents=True, exist_ok=True)
    with JUDGED_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Build report
    report = {
        "total_records": n_total,
        "agreed": n_agreed,
        "corrected": n_corrected,
        "missing_judgment": n_missing_judgment,
        "agreement_rate": round(n_agreed / max(1, n_total - n_missing_judgment), 4),
        "by_aspect": {
            asp: {
                "total": d["total"],
                "agreed": d["agreed"],
                "agreement_rate": round(d["agreed"] / max(1, d["total"]), 4),
            }
            for asp, d in sorted(per_aspect.items(), key=lambda x: -x[1]["total"])
        },
        "top_correction_patterns": correction_patterns.most_common(20),
        "sample_correction_notes": {
            asp: notes[:5] for asp, notes in notes_by_aspect.items()
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total:              {n_total}")
    print(f"  Agreed:           {n_agreed}")
    print(f"  Corrected:        {n_corrected}")
    print(f"  Missing judgment: {n_missing_judgment}")
    print(f"  Agreement rate:   {report['agreement_rate']:.2%}")
    print(f"\nBy aspect:")
    for asp, d in report["by_aspect"].items():
        print(f"  {asp:20s} {d['agreed']:3d}/{d['total']:3d}  ({d['agreement_rate']:.2%})")
    print(f"\nJudged CSV: {JUDGED_CSV}")
    print(f"Report:     {REPORT_PATH}")


if __name__ == "__main__":
    main()

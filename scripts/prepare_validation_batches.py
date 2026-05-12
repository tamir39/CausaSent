"""Split validation_sample.csv into N JSON batches for parallel human-judging subagents."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

VAL_CSV = Path("data/interim/validation_sample.csv")
BATCH_DIR = Path("data/interim/validation_batches")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=50)
    args = ap.parse_args()

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    with VAL_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    n = args.batch_size
    batches = [rows[i:i + n] for i in range(0, len(rows), n)]
    for i, batch in enumerate(batches, start=1):
        (BATCH_DIR / f"val_batch_{i:03d}.json").write_text(
            json.dumps(batch, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"{len(rows)} rows → {len(batches)} batches of up to {n}")


if __name__ == "__main__":
    main()

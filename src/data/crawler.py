"""Raw-review collection.

We deliberately do NOT ship a Shopee / TikTok Shop scraper here — both ToS-forbid
automated collection and aggressively block clients. The intended workflow is:

1. Bootstrap from public Vietnamese review datasets (e.g. UIT-VSFC, ViSFD,
   Foody crawls released by NLP groups). Place them under data/raw/ as JSONL
   with `{"id": ..., "review": ...}` per line.
2. For diversity, do a small *manual-export* pass: open product pages in a
   browser, copy reviews, and append to the JSONL. This stays within ToS and
   is enough for the targeted-crawl supplement called for in SPEC §1.1.

This module exposes a single helper that normalizes any of the above sources
into the JSONL format the rest of the pipeline expects.
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path


def to_jsonl(rows: list[dict], out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            if "id" not in r:
                r["id"] = uuid.uuid4().hex[:12]
            if "review" not in r:
                raise ValueError(f"row missing 'review': {r}")
            f.write(json.dumps({"id": r["id"], "review": r["review"]}, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize a CSV/JSON of raw reviews to JSONL.")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--text-key", default="review", help="Field name holding the review text.")
    args = ap.parse_args()

    p = Path(args.in_path)
    if p.suffix.lower() == ".csv":
        import csv

        with open(p, "r", encoding="utf-8", newline="") as f:
            rows = [{"review": row[args.text_key]} for row in csv.DictReader(f) if row.get(args.text_key)]
    else:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = [{"review": r[args.text_key]} for r in data if r.get(args.text_key)]

    to_jsonl(rows, args.out_path)
    print(f"[crawler] wrote {len(rows)} rows → {args.out_path}")


if __name__ == "__main__":
    main()

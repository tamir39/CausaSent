"""Verify a batch annotation JSONL — checks review[start:end] == aspect_term."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="Input batch JSON (records with review)")
    ap.add_argument("--annotated", required=True, help="Annotated JSONL output")
    args = ap.parse_args()

    batch = {r["key"]: r for r in json.loads(Path(args.batch).read_text(encoding="utf-8-sig"))}
    annotated_lines = Path(args.annotated).read_text(encoding="utf-8-sig").splitlines()

    n_ok = 0
    n_bad = 0
    n_missing = 0
    bad: list[tuple[str, str, str]] = []
    seen_keys: set[str] = set()

    for line in annotated_lines:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        key = obj["key"]
        seen_keys.add(key)
        if key not in batch:
            n_missing += 1
            bad.append((key, "(missing in input)", ""))
            continue
        review = batch[key]["review"]
        s, e = obj["aspect_term_span"]
        term = obj["aspect_term"]
        got = review[s:e] if 0 <= s < len(review) and 0 <= e <= len(review) else "(out of range)"
        if got == term:
            n_ok += 1
        else:
            n_bad += 1
            bad.append((key, term, got))

    missing_from_output = set(batch.keys()) - seen_keys

    print(f"OK:                {n_ok}")
    print(f"BAD span:          {n_bad}")
    print(f"Missing in input:  {n_missing}")
    print(f"Missing in output: {len(missing_from_output)}")

    if bad:
        print("\nFirst 20 bad:")
        for k, t, g in bad[:20]:
            print(f"  {k}: term='{t}' got='{g}'")

    if missing_from_output:
        print("\nFirst 20 missing-in-output keys:")
        for k in list(missing_from_output)[:20]:
            print(f"  {k}")

    if n_bad or missing_from_output:
        sys.exit(1)


if __name__ == "__main__":
    main()

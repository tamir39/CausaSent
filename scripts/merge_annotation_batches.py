"""
Merge all annotated batch outputs into a single draft JSONL.

Usage:
    python scripts/merge_annotation_batches.py [--source gold|tiki] [--out PATH]

Defaults to gold. Selecting tiki reads from data/interim/tiki_batches/ and writes
to data/interim/tiki_aspect_term_draft.jsonl.

Each output line:
    {
      "id": "vlsp2018-08975" or "tiki-14357443",
      "ann_idx": 0,
      "review": "...",
      "aspect_term": "Quán",
      "aspect_term_span": [0, 4],
      "aspect_category": "appearance",
      "sentiment": "positive",
      "valid": true,
      "source_batch": 1,
      "source": "gold" | "tiki"
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCES = {
    "gold": {
        "batch_dir": Path("data/interim/batches"),
        "draft_path": Path("data/interim/gold_aspect_term_draft.jsonl"),
    },
    "tiki": {
        "batch_dir": Path("data/interim/tiki_batches"),
        "draft_path": Path("data/interim/tiki_aspect_term_draft.jsonl"),
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), default="gold",
                    help="Which annotation pool to merge")
    ap.add_argument("--out", default=None,
                    help="Override output draft path")
    args = ap.parse_args()

    cfg = SOURCES[args.source]
    batch_dir: Path = cfg["batch_dir"]
    out_dir = batch_dir / "output"
    draft_path = Path(args.out) if args.out else cfg["draft_path"]

    manifest_path = batch_dir / "MANIFEST.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found.")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batches = manifest["batches"]

    total_in = 0
    total_out = 0
    total_bad = 0
    missing_files: list[str] = []

    draft_path.parent.mkdir(parents=True, exist_ok=True)
    with draft_path.open("w", encoding="utf-8") as fout:
        for i, entry in enumerate(batches, start=1):
            in_path = batch_dir / entry["batch"]
            out_path = out_dir / entry["batch"].replace(".json", "_annotated.jsonl")
            if not out_path.exists():
                missing_files.append(out_path.name)
                continue

            in_records = {r["key"]: r for r in json.loads(in_path.read_text(encoding="utf-8-sig"))}
            total_in += len(in_records)

            for line in out_path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line:
                    continue
                ann = json.loads(line)
                key = ann["key"]
                if key not in in_records:
                    total_bad += 1
                    continue
                src = in_records[key]
                review = src["review"]
                s, e = ann["aspect_term_span"]
                term = ann["aspect_term"]

                # Final verification — drop if span doesn't match
                if not (0 <= s < len(review) and 0 < e <= len(review) and review[s:e] == term):
                    total_bad += 1
                    continue

                # Split key back into id + ann_idx
                rid, _, idx = key.rpartition("_")

                fout.write(json.dumps({
                    "id": rid,
                    "ann_idx": int(idx),
                    "review": review,
                    "aspect_term": term,
                    "aspect_term_span": [s, e],
                    "aspect_category": ann["aspect_category"],
                    "sentiment": ann["sentiment"],
                    "valid": True,
                    "source_batch": i,
                    "source": args.source,
                }, ensure_ascii=False) + "\n")
                total_out += 1

    print(f"Source:           {args.source}")
    print(f"Input records:    {total_in}")
    print(f"Output (valid):   {total_out}")
    print(f"Dropped (bad):    {total_bad}")
    print(f"Missing batches:  {len(missing_files)}")
    if missing_files:
        print("  " + "\n  ".join(missing_files))
    print(f"\nDraft written to: {draft_path}")


if __name__ == "__main__":
    main()

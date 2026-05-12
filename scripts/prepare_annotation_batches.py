"""
Split CausaSent gold annotations (non-neutral) into batched JSON files for
parallel Claude-based annotation via subagents.

Output:
    data/interim/batches/batch_NNN.json   — input records per batch
    data/interim/batches/MANIFEST.json    — listing of all batches

Each batch record:
    {
        "key": "vlsp2018-08975_0",      # id + ann_idx, unique
        "review": "<full review text>",
        "aspect_category": "appearance",
        "sentiment": "positive",
        "cause_text": "Quán nhỏ xinh"
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GOLD_SPLITS = [
    "data/gold/train.json",
    "data/gold/val.json",
    "data/gold/test.json",
]
BATCH_DIR = Path("data/interim/batches")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=100,
                    help="Records per batch (default 100)")
    args = ap.parse_args()

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    flat: list[dict] = []
    for split_path in GOLD_SPLITS:
        path = Path(split_path)
        if not path.exists():
            print(f"WARNING: {split_path} missing")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for r in data:
            review = r["review"]
            rid = r["id"]
            for idx, ann in enumerate(r.get("annotations", [])):
                if ann["sentiment"] == "neutral":
                    continue
                flat.append({
                    "key": f"{rid}_{idx}",
                    "review": review,
                    "aspect_category": ann["aspect"],
                    "sentiment": ann["sentiment"],
                    "cause_text": ann.get("cause_text", ""),
                })

    print(f"Flattened {len(flat)} non-neutral annotations")

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
    print(f"Manifest: {BATCH_DIR}/MANIFEST.json")


if __name__ == "__main__":
    main()

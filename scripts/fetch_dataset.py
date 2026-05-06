"""Fetch the CausaSent gold dataset from HuggingFace Hub.

Pulls ``Tamir39/causasent`` and materializes it under ``<dest>``:

    <dest>/
      gold/
        train.json
        val.json
        test.json
      processed/    (optional; only if pseudo-labels were uploaded)
        weak.json

After this, training configs (`configs/phobert.yaml`, `configs/mt5.yaml`)
that reference `data/processed/*` and `data/gold/*` work directly when
`<dest>` is the repo `data/` directory.

Auth: a public dataset doesn't strictly require a token. For private repos
or rate-limit avoidance, set `HF_TOKEN` in the env (Kaggle: Add-ons →
Secrets → HF_TOKEN).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_REPO_ID = "Tamir39/causasent"
DEFAULT_DEST = Path("data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--dest",
        default=os.environ.get("CAUSASENT_DATA_DIR") or str(DEFAULT_DEST),
        help=f"destination directory (default: $CAUSASENT_DATA_DIR or '{DEFAULT_DEST}')",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: pip install -r requirements.txt")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.repo_id} -> {dest}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        force_download=args.force,
        token=os.environ.get("HF_TOKEN"),
        max_workers=4,
    )

    json_files = sorted(p for p in dest.rglob("*.json") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"done: {len(json_files)} JSON files, {total_bytes / 1024:.1f} KB at {dest}")
    for p in json_files[:10]:
        print(f"  {p.relative_to(dest)}")
    if len(json_files) > 10:
        print(f"  … and {len(json_files) - 10} more")


if __name__ == "__main__":
    main()

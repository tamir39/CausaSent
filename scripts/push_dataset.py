"""Push the CausaSent gold dataset to HuggingFace Hub.

Default repo: ``Tamir39/causasent`` (dataset, public, CC-BY-SA-4.0).

Uploads the contents of ``data/`` (or ``--source``) — typically::

    data/
      gold/{train,val,test}.json
      processed/weak.json   (optional)
      HF_README.md          (rendered as the dataset card if present)

Auth: ``hf auth login`` once locally, or set ``HF_TOKEN`` in the env.

Usage:
    python scripts/push_dataset.py
    python scripts/push_dataset.py --source data --repo-id Tamir39/causasent
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = "Tamir39/causasent"
DEFAULT_SOURCE = ROOT / "data"
HF_README = ROOT / "data" / "HF_README.md"
IGNORE = ["**/__pycache__/**", "**/raw/**", ".DS_Store", "*.tmp", ".gitkeep"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true", help="create as a private dataset repo")
    args = parser.parse_args()

    src: Path = args.source
    if not src.is_dir():
        raise FileNotFoundError(f"source dir not found: {src}")

    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    print(f"whoami: {api.whoami()['name']}")

    create_repo(args.repo_id, repo_type="dataset", exist_ok=True, private=args.private)
    print(f"repo ready: https://huggingface.co/datasets/{args.repo_id}")

    if HF_README.is_file():
        print("  uploading dataset card -> README.md")
        api.upload_file(
            path_or_fileobj=str(HF_README),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="upload dataset card",
        )

    print(f"  uploading {src} -> /")
    api.upload_folder(
        folder_path=str(src),
        repo_id=args.repo_id,
        repo_type="dataset",
        ignore_patterns=IGNORE,
        commit_message="upload CausaSent gold dataset",
    )
    print(f"\ndone -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

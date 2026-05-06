"""Split a labeled JSON file into train/val/test (80/10/10) by review id."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.data.schema import load_reviews, save_reviews


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    reviews = load_reviews(args.in_path)
    rng = random.Random(args.seed)
    rng.shuffle(reviews)

    n = len(reviews)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train, val, test = reviews[:n_train], reviews[n_train:n_train + n_val], reviews[n_train + n_val:]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_reviews(train, out / "train.json")
    save_reviews(val, out / "val.json")
    save_reviews(test, out / "test.json")
    print(f"split: train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()

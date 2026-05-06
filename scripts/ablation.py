"""Minimal ablation: train PhoBERT on (a) gold-only and (b) gold+validated-pseudo,
then compare entity-F1 on the same test split.

Usage:
    python scripts/ablation.py \\
        --base-config configs/phobert.yaml \\
        --gold-train data/gold/tagging_train.json \\
        --pseudo-train data/processed/tagging_pseudo.json \\
        --val data/gold/tagging_val.json \\
        --test data/gold/tagging_test.json \\
        --out runs/ablation

The script writes a small `summary.json` with both runs' F1 numbers.
Heavy lifting (training, eval) reuses src.train.train_phobert and
src.eval.eval_phobert — this script just orchestrates and merges JSON.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from src.data.schema import load_reviews, save_reviews
from src.eval.eval_phobert import evaluate


def _merge_json(a: str, b: str, out: str) -> None:
    aa = load_reviews(a)
    bb = load_reviews(b)
    save_reviews(aa + bb, out)


def _write_config(base_cfg: dict, train_path: str, val_path: str, test_path: str,
                  ckpt_dir: str, log_dir: str, out_path: str) -> None:
    cfg = json.loads(json.dumps(base_cfg))  # deep copy via JSON
    cfg["data"]["train_path"] = train_path
    cfg["data"]["val_path"] = val_path
    cfg["data"]["test_path"] = test_path
    cfg["output"]["ckpt_dir"] = ckpt_dir
    cfg["output"]["log_dir"] = log_dir
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def _train(config_path: str) -> None:
    subprocess.check_call([sys.executable, "-m", "src.train.train_phobert", "--config", config_path])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--gold-train", required=True)
    ap.add_argument("--pseudo-train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="runs/ablation")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(open(args.base_config, "r", encoding="utf-8"))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Run A — gold only.
    a_dir = out_root / "gold_only"
    a_dir.mkdir(parents=True, exist_ok=True)
    a_cfg_path = str(a_dir / "config.yaml")
    _write_config(base_cfg, args.gold_train, args.val, args.test,
                  ckpt_dir=str(a_dir / "ckpt"), log_dir=str(a_dir / "logs"),
                  out_path=a_cfg_path)
    _train(a_cfg_path)
    a_metrics = evaluate(str(Path(a_dir / "ckpt" / "best.pt")), a_cfg_path, split="test")

    # Run B — gold + pseudo.
    b_dir = out_root / "gold_plus_pseudo"
    b_dir.mkdir(parents=True, exist_ok=True)
    merged_train = str(b_dir / "train_merged.json")
    _merge_json(args.gold_train, args.pseudo_train, merged_train)
    b_cfg_path = str(b_dir / "config.yaml")
    _write_config(base_cfg, merged_train, args.val, args.test,
                  ckpt_dir=str(b_dir / "ckpt"), log_dir=str(b_dir / "logs"),
                  out_path=b_cfg_path)
    _train(b_cfg_path)
    b_metrics = evaluate(str(Path(b_dir / "ckpt" / "best.pt")), b_cfg_path, split="test")

    summary = {
        "gold_only": a_metrics,
        "gold_plus_pseudo": b_metrics,
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

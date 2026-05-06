"""Sanity-check script for the CausaSent dev / Kaggle environment.

Verifies GPU access, library imports, and config loading. Robust to single
failures — keeps going and reports a final summary.
"""
from __future__ import annotations

import os
import sys

# Pin a non-interactive matplotlib backend. Some Kaggle kernels export
# MPLBACKEND=module://matplotlib_inline.backend_inline; subprocess venvs
# don't have matplotlib_inline, so any pyplot import blows up. Agg is always
# available.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["MPLBACKEND"] = "Agg"

import torch
import yaml


LIBRARIES = [
    "torch", "transformers", "sentencepiece", "accelerate", "datasets",
    "pydantic", "yaml", "tqdm", "numpy", "pandas", "sklearn",
    "seqeval", "rouge_score", "gradio", "huggingface_hub",
]

CONFIGS = ["configs/phobert.yaml", "configs/mt5.yaml"]


def check_env() -> int:
    print("--- CausaSent environment sanity check ---")

    print("\n[1/3] GPU")
    gpu = torch.cuda.is_available()
    print(f"  CUDA available: {gpu}")
    if gpu:
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("  (training will run on CPU — slow)")

    print("\n[2/3] Library imports")
    failed: list[str] = []
    for lib in LIBRARIES:
        try:
            __import__(lib)
            print(f"  [OK]   {lib}")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {lib} — {type(e).__name__}: {e}")
            failed.append(lib)

    print("\n[3/3] Configs")
    config_ok = True
    for cfg_path in CONFIGS:
        if not os.path.exists(cfg_path):
            print(f"  [FAIL] {cfg_path} not found")
            config_ok = False
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            for k in ("model", "data", "train", "output"):
                if k not in cfg:
                    print(f"  [WARN] {cfg_path} missing key: {k}")
            print(f"  [OK]   {cfg_path}")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {cfg_path} — {e}")
            config_ok = False

    print("\n" + "=" * 40)
    if not failed and config_ok and gpu:
        print("Conclusion: ready to train.")
        return 0
    bits = []
    if not gpu:
        bits.append("no GPU")
    if failed:
        bits.append(f"{len(failed)} import(s) failed: {failed}")
    if not config_ok:
        bits.append("config issue(s)")
    print("Conclusion: incomplete — " + "; ".join(bits))
    print("=" * 40)
    return 1


if __name__ == "__main__":
    sys.exit(check_env())

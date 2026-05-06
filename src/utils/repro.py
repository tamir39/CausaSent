"""Reproducibility helpers — seeding and run-metadata dumping."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _env_summary() -> str:
    lines = [
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
    ]
    for pkg in ("torch", "transformers", "pydantic", "seqeval"):
        try:
            mod = __import__(pkg)
            lines.append(f"{pkg}: {getattr(mod, '__version__', '?')}")
        except ImportError:
            lines.append(f"{pkg}: <not installed>")
    try:
        import torch
        lines.append(f"cuda: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"cuda_device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass
    return "\n".join(lines) + "\n"


def make_run_dir(base: str | Path = "runs", tag: str = "") -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{ts}-{tag}" if tag else ts
    out = Path(base) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def dump_run_metadata(run_dir: str | Path, cfg: dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    with open(run_dir / "git_sha.txt", "w", encoding="utf-8") as f:
        f.write(_git_sha() + "\n")
    with open(run_dir / "env.txt", "w", encoding="utf-8") as f:
        f.write(_env_summary())
    # Touch metrics.jsonl so log_metrics can append.
    (run_dir / "metrics.jsonl").touch()
    return run_dir


def log_metrics(run_dir: str | Path, record: dict[str, Any]) -> None:
    path = Path(run_dir) / "metrics.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

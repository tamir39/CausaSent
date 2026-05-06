"""Train the PhoBERT two-head tagger."""
from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ..data.dataset import load_tagging_dataset
from ..data.label_schema import (
    ASPECT_SENT_LABELS,
    CAUSE_LABELS,
    IGNORE_INDEX,
)
from ..models.phobert_tagger import PhoBertTwoHeadTagger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _inverse_freq_weights(dataset, n_classes: int, label_key: str) -> torch.Tensor:
    counts = Counter()
    for i in range(len(dataset)):
        labels = dataset[i][label_key].tolist()
        counts.update(l for l in labels if l != IGNORE_INDEX)
    weights = torch.ones(n_classes, dtype=torch.float)
    total = sum(counts.values()) or 1
    for c in range(n_classes):
        cnt = counts.get(c, 0)
        weights[c] = total / (n_classes * (cnt + 1))
    return weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = PhoBertTwoHeadTagger.load_tokenizer(cfg["model"]["pretrained"])
    seg_kind = cfg["data"].get("word_segmenter", "vncorenlp")

    train_ds = load_tagging_dataset(cfg["data"]["train_path"], tokenizer, cfg["data"]["max_len"], seg_kind)
    val_ds = load_tagging_dataset(cfg["data"]["val_path"], tokenizer, cfg["data"]["max_len"], seg_kind)

    asp_w = None
    cau_w = None
    if cfg["train"].get("class_weighting") == "inverse_freq":
        asp_w = _inverse_freq_weights(train_ds, len(ASPECT_SENT_LABELS), "asp_labels").to(device)
        cau_w = _inverse_freq_weights(train_ds, len(CAUSE_LABELS), "cause_labels").to(device)

    model = PhoBertTwoHeadTagger(
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
        asp_class_weights=asp_w,
        cause_class_weights=cau_w,
        cause_loss_weight=cfg["train"]["cause_loss_weight"],
    ).to(device)

    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    epochs = cfg["train"]["epochs"]
    total_steps = max(1, len(train_loader) * epochs)
    optim = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=cfg["train"]["weight_decay"],
    )
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(total_steps * cfg["train"]["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    ckpt_dir = Path(cfg["output"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = math.inf

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            optim.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optim.step()
            sched.step()
            running += out.loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss={running / step:.4f}")

        # Validation.
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                v_loss += model(**batch).loss.item()
        v_loss /= max(1, len(val_loader))
        print(f"[epoch {epoch}] val_loss={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            torch.save({"model": model.state_dict(), "config": cfg}, ckpt_dir / "best.pt")
            print(f"  ↳ saved best checkpoint (val_loss={v_loss:.4f})")

    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_dir / "last.pt")


if __name__ == "__main__":
    main()

"""Train the PhoBERT two-head tagger.

Best checkpoint is selected by **entity-level F1** on the val split (mean of
aspect-sentiment F1 and cause F1), not by validation loss. Loss and F1 are
weakly correlated; F1 is what we actually care about downstream.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ..data.dataset import load_tagging_dataset
from ..data.label_schema import (
    ASPECT_SENT_ID2LABEL,
    ASPECT_SENT_LABELS,
    CAUSE_ID2LABEL,
    CAUSE_LABELS,
    IGNORE_INDEX,
)
from ..inference.decode import _bio_segments
from ..models.phobert_tagger import PhoBertTwoHeadTagger
from ..utils.repro import dump_run_metadata, log_metrics, make_run_dir, set_seed


def _inverse_freq_weights(dataset, n_classes: int, label_key: str) -> torch.Tensor:
    counts: Counter = Counter()
    for i in range(len(dataset)):
        labels = dataset[i][label_key].tolist()
        counts.update(l for l in labels if l != IGNORE_INDEX)
    weights = torch.ones(n_classes, dtype=torch.float)
    total = sum(counts.values()) or 1
    for c in range(n_classes):
        cnt = counts.get(c, 0)
        weights[c] = total / (n_classes * (cnt + 1))
    return weights


def _entity_f1(pred_set: set, gold_set: set) -> float:
    tp = len(pred_set & gold_set)
    p = tp / max(1, len(pred_set))
    r = tp / max(1, len(gold_set))
    return 2 * p * r / max(1e-9, p + r)


@torch.no_grad()
def _eval_entity_f1(model, val_loader, device) -> tuple[float, float, float]:
    """Return (asp_f1, cause_f1, mean_f1) on the val loader."""
    model.eval()
    asp_pred: set = set()
    asp_gold: set = set()
    cau_pred: set = set()
    cau_gold: set = set()
    doc_id = 0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"], batch["attention_mask"])
        ap = out.asp_logits.argmax(-1).cpu().tolist()
        cp = out.cause_logits.argmax(-1).cpu().tolist()
        ag = batch["asp_labels"].cpu().tolist()
        cg = batch["cause_labels"].cpu().tolist()
        for j in range(len(ap)):
            ap_tags = [ASPECT_SENT_ID2LABEL[p] for p, g in zip(ap[j], ag[j]) if g != IGNORE_INDEX]
            ag_tags = [ASPECT_SENT_ID2LABEL[g] for g in ag[j] if g != IGNORE_INDEX]
            cp_tags = [CAUSE_ID2LABEL[p] for p, g in zip(cp[j], cg[j]) if g != IGNORE_INDEX]
            cg_tags = [CAUSE_ID2LABEL[g] for g in cg[j] if g != IGNORE_INDEX]
            for s, e, base in _bio_segments(ap_tags):
                asp_pred.add((doc_id, s, e, base))
            for s, e, base in _bio_segments(ag_tags):
                asp_gold.add((doc_id, s, e, base))
            for s, e, _b in _bio_segments(cp_tags):
                cau_pred.add((doc_id, s, e))
            for s, e, _b in _bio_segments(cg_tags):
                cau_gold.add((doc_id, s, e))
            doc_id += 1
    asp_f1 = _entity_f1(asp_pred, asp_gold)
    cau_f1 = _entity_f1(cau_pred, cau_gold)
    return asp_f1, cau_f1, (asp_f1 + cau_f1) / 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = make_run_dir(cfg["output"].get("log_dir", "runs"), tag="phobert")
    dump_run_metadata(run_dir, cfg)

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
    best_f1 = -1.0

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

        # Validation: loss for monitoring, F1 for checkpoint selection.
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                v_loss += model(**batch).loss.item()
        v_loss /= max(1, len(val_loader))

        asp_f1, cau_f1, mean_f1 = _eval_entity_f1(model, val_loader, device)
        print(f"[epoch {epoch}] val_loss={v_loss:.4f} asp_f1={asp_f1:.4f} cause_f1={cau_f1:.4f} mean_f1={mean_f1:.4f}")
        log_metrics(run_dir, {
            "epoch": epoch, "val_loss": v_loss,
            "asp_f1": asp_f1, "cause_f1": cau_f1, "mean_f1": mean_f1,
        })

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            torch.save({"model": model.state_dict(), "config": cfg, "mean_f1": mean_f1}, ckpt_dir / "best.pt")
            print(f"  ↳ saved best checkpoint (mean_f1={mean_f1:.4f})")

    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_dir / "last.pt")
    log_metrics(run_dir, {"event": "done", "best_mean_f1": best_f1})


if __name__ == "__main__":
    main()

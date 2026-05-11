"""Train the PhoBERT ABSA model (two-head: ATE + binary sentiment).

Best checkpoint is selected by entity-level F1 on the val split:
  mean of ATE-category F1 (span + category match) and Sentiment Accuracy at B-tokens.
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
    ATE_ID2LABEL,
    ATE_LABELS,
    IGNORE_INDEX,
    SENTIMENT_ID2LABEL,
    SENTIMENT_LABELS,
)
from ..inference.decode import _bio_segments
from ..models.phobert_tagger import PhoBertABSA
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


def _sent_accuracy(pred_list: list[int], gold_list: list[int]) -> float:
    if not gold_list:
        return 0.0
    correct = sum(p == g for p, g in zip(pred_list, gold_list))
    return correct / len(gold_list)


@torch.no_grad()
def _eval(model, val_loader, device) -> tuple[float, float, float]:
    """Return (ate_f1, sent_acc, mean_metric) on the val loader.

    ate_f1: entity-level F1 on (aspect_category, start, end) tuples
    sent_acc: accuracy of sentiment prediction at B-token positions only
    """
    model.eval()
    ate_pred: set = set()
    ate_gold: set = set()
    sent_pred_all: list[int] = []
    sent_gold_all: list[int] = []
    doc_id = 0

    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"], batch["attention_mask"])
        ap = out.ate_logits.argmax(-1).cpu().tolist()
        sp = out.sent_logits.argmax(-1).cpu().tolist()
        ag = batch["ate_labels"].cpu().tolist()
        sg = batch["sent_labels"].cpu().tolist()

        for j in range(len(ap)):
            # ATE entity F1 — only on non-IGNORE positions
            ap_tags = [ATE_ID2LABEL.get(p, "O") for p, g in zip(ap[j], ag[j]) if g != IGNORE_INDEX]
            ag_tags = [ATE_ID2LABEL.get(g, "O") for g in ag[j] if g != IGNORE_INDEX]
            for s, e, base in _bio_segments(ap_tags):
                ate_pred.add((doc_id, s, e, base))
            for s, e, base in _bio_segments(ag_tags):
                ate_gold.add((doc_id, s, e, base))

            # Sentiment accuracy — only at B-token positions (where gold != IGNORE_INDEX)
            for pred_s, gold_s in zip(sp[j], sg[j]):
                if gold_s == IGNORE_INDEX:
                    continue
                sent_pred_all.append(pred_s)
                sent_gold_all.append(gold_s)

            doc_id += 1

    ate_f1 = _entity_f1(ate_pred, ate_gold)
    sent_acc = _sent_accuracy(sent_pred_all, sent_gold_all)
    mean_metric = (ate_f1 + sent_acc) / 2.0
    return ate_f1, sent_acc, mean_metric


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = make_run_dir(cfg["output"].get("log_dir", "runs"), tag="phobert")
    dump_run_metadata(run_dir, cfg)

    tokenizer = PhoBertABSA.load_tokenizer(cfg["model"]["pretrained"])
    seg_kind = cfg["data"].get("word_segmenter", "vncorenlp")

    train_ds = load_tagging_dataset(cfg["data"]["train_path"], tokenizer, cfg["data"]["max_len"], seg_kind)
    val_ds = load_tagging_dataset(cfg["data"]["val_path"], tokenizer, cfg["data"]["max_len"], seg_kind)

    ate_w = None
    sent_w = None
    if cfg["train"].get("class_weighting") == "inverse_freq":
        ate_w = _inverse_freq_weights(train_ds, len(ATE_LABELS), "ate_labels").to(device)
        sent_w = _inverse_freq_weights(train_ds, len(SENTIMENT_LABELS), "sent_labels").to(device)

    model = PhoBertABSA(
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
        ate_class_weights=ate_w,
        sent_class_weights=sent_w,
        sent_loss_weight=cfg["train"].get("sent_loss_weight", 1.0),
        contrastive_weight=cfg["train"].get("contrastive_weight", 0.0),
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
    best_metric = -1.0

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

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                v_loss += model(**batch).loss.item()
        v_loss /= max(1, len(val_loader))

        ate_f1, sent_acc, mean_metric = _eval(model, val_loader, device)
        print(
            f"[epoch {epoch}] val_loss={v_loss:.4f} "
            f"ate_f1={ate_f1:.4f} sent_acc={sent_acc:.4f} mean={mean_metric:.4f}"
        )
        log_metrics(run_dir, {
            "epoch": epoch, "val_loss": v_loss,
            "ate_f1": ate_f1, "sent_acc": sent_acc, "mean_metric": mean_metric,
        })

        if mean_metric > best_metric:
            best_metric = mean_metric
            torch.save(
                {"model": model.state_dict(), "config": cfg, "mean_metric": mean_metric},
                ckpt_dir / "best.pt",
            )
            print(f"  saved best checkpoint (mean={mean_metric:.4f})")

    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_dir / "last.pt")
    log_metrics(run_dir, {"event": "done", "best_mean_metric": best_metric})


if __name__ == "__main__":
    main()

"""Evaluation for the PhoBERT ABSA model (ATE BIO + binary sentiment).

Metrics:
- Token-level seqeval F1 on the ATE BIO tags (per aspect_category).
- Entity-level F1 on (start_word, end_word, aspect_category) tuples.
- Sentiment metrics at B-token positions: accuracy + binary F1 (positive vs negative).

Usage:
    python -m src.eval.eval_phobert \
        --config configs/phobert.yaml \
        --ckpt checkpoints/phobert/best.pt \
        --split test
"""
from __future__ import annotations

import argparse

import torch
import yaml
from seqeval.metrics import classification_report as seqeval_report
from sklearn.metrics import classification_report as sk_report, f1_score
from torch.utils.data import DataLoader

from ..data.dataset import load_tagging_dataset
from ..data.label_schema import (
    ATE_ID2LABEL,
    IGNORE_INDEX,
    SENTIMENT_ID2LABEL,
)
from ..inference.decode import _bio_segments
from ..models.phobert_tagger import PhoBertABSA


def _prf(preds: set, golds: set) -> tuple[float, float, float]:
    tp = len(preds & golds)
    p = tp / max(1, len(preds))
    r = tp / max(1, len(golds))
    f = 2 * p * r / max(1e-9, p + r)
    return p, r, f


def evaluate(ckpt_path: str, config_path: str, split: str = "test") -> dict:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = PhoBertABSA.load_tokenizer(cfg["model"]["pretrained"])
    seg_kind = cfg["data"].get("word_segmenter", "vncorenlp")

    path_key = {"train": "train_path", "val": "val_path", "test": "test_path"}[split]
    ds = load_tagging_dataset(cfg["data"][path_key], tokenizer, cfg["data"]["max_len"], seg_kind)

    model = PhoBertABSA(pretrained=cfg["model"]["pretrained"]).to(device)
    state = torch.load(ckpt_path, map_location=device)
    # strict=False: training saves class-weight tensors into the loss modules
    # that aren't reconstructed at eval time.
    model.load_state_dict(state["model"] if "model" in state else state, strict=False)
    model.eval()

    ate_pred_tags: list[list[str]] = []
    ate_gold_tags: list[list[str]] = []
    ate_pred_ent: set[tuple[int, int, int, str]] = set()
    ate_gold_ent: set[tuple[int, int, int, str]] = set()
    sent_preds: list[int] = []
    sent_golds: list[int] = []

    loader = DataLoader(ds, batch_size=8, shuffle=False)
    with torch.no_grad():
        b_offset = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"])
            ap = out.ate_logits.argmax(-1).cpu().tolist()
            sp = out.sent_logits.argmax(-1).cpu().tolist()
            ag = batch["ate_labels"].cpu().tolist()
            sg = batch["sent_labels"].cpu().tolist()

            for j in range(len(ap)):
                pred_tags = [
                    ATE_ID2LABEL.get(p, "O")
                    for p, g in zip(ap[j], ag[j]) if g != IGNORE_INDEX
                ]
                gold_tags = [
                    ATE_ID2LABEL.get(g, "O")
                    for g in ag[j] if g != IGNORE_INDEX
                ]
                ate_pred_tags.append(pred_tags)
                ate_gold_tags.append(gold_tags)

                doc_id = b_offset + j
                for s, e, base in _bio_segments(pred_tags):
                    ate_pred_ent.add((doc_id, s, e, base))
                for s, e, base in _bio_segments(gold_tags):
                    ate_gold_ent.add((doc_id, s, e, base))

                for pred_s, gold_s in zip(sp[j], sg[j]):
                    if gold_s == IGNORE_INDEX:
                        continue
                    sent_preds.append(pred_s)
                    sent_golds.append(gold_s)
            b_offset += len(ap)

    print("=== ATE BIO (token-level seqeval) ===")
    print(seqeval_report(ate_gold_tags, ate_pred_tags, digits=4))

    p_a, r_a, f_a = _prf(ate_pred_ent, ate_gold_ent)
    print("=== ATE entity-level (start, end, aspect_category) ===")
    print(f"  P={p_a:.4f}  R={r_a:.4f}  F1={f_a:.4f}")

    print("=== Sentiment (at B-token positions) ===")
    if sent_golds:
        sent_acc = sum(p == g for p, g in zip(sent_preds, sent_golds)) / len(sent_golds)
        sent_f1_macro = f1_score(sent_golds, sent_preds, average="macro")
        sent_f1_binary = f1_score(sent_golds, sent_preds, average="binary", pos_label=1)
        target_names = [SENTIMENT_ID2LABEL[i] for i in sorted(SENTIMENT_ID2LABEL)]
        print(sk_report(sent_golds, sent_preds, target_names=target_names, digits=4))
        print(f"  accuracy={sent_acc:.4f}  macro_f1={sent_f1_macro:.4f}  binary_f1(neg)={sent_f1_binary:.4f}")
    else:
        sent_acc = sent_f1_macro = sent_f1_binary = 0.0
        print("  (no sentiment labels evaluated)")

    return {
        "ate_entity": {"p": p_a, "r": r_a, "f1": f_a},
        "sentiment": {
            "accuracy": sent_acc,
            "macro_f1": sent_f1_macro,
            "binary_f1_negative": sent_f1_binary,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()
    evaluate(args.ckpt, args.config, args.split)


if __name__ == "__main__":
    main()

"""Token-level + entity-level F1 for the PhoBERT tagger.

Entity-level F1 is computed *per head*:
- Aspect-sentiment head: an entity is (start_word, end_word, aspect, sentiment).
- Cause head:           an entity is (start_word, end_word).

We use seqeval for token-level F1 (it operates on BIO sequences directly).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from seqeval.metrics import classification_report as seqeval_report
from torch.utils.data import DataLoader

from ..data.dataset import load_tagging_dataset
from ..data.label_schema import (
    ASPECT_SENT_ID2LABEL,
    CAUSE_ID2LABEL,
    IGNORE_INDEX,
)
from ..inference.decode import _bio_segments
from ..models.phobert_tagger import PhoBertTwoHeadTagger


def _word_level_tags(label_ids: list[int], id2label: dict[int, str]) -> list[str]:
    return [id2label[i] for i in label_ids if i != IGNORE_INDEX]


def _entity_set(tags: list[str]) -> set[tuple[int, int, str]]:
    return {(s, e, base) for (s, e, base) in _bio_segments(tags)}


def _prf(preds: set, golds: set) -> tuple[float, float, float]:
    tp = len(preds & golds)
    p = tp / max(1, len(preds))
    r = tp / max(1, len(golds))
    f = 2 * p * r / max(1e-9, p + r)
    return p, r, f


def evaluate(ckpt_path: str, config_path: str, split: str = "test") -> dict:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = PhoBertTwoHeadTagger.load_tokenizer(cfg["model"]["pretrained"])
    seg_kind = cfg["data"].get("word_segmenter", "vncorenlp")

    path_key = {"train": "train_path", "val": "val_path", "test": "test_path"}[split]
    ds = load_tagging_dataset(cfg["data"][path_key], tokenizer, cfg["data"]["max_len"], seg_kind)

    model = PhoBertTwoHeadTagger(pretrained=cfg["model"]["pretrained"]).to(device)
    state = torch.load(ckpt_path, map_location=device)
    # strict=False: training saves asp_loss_fn.weight / cause_loss_fn.weight
    # (inverse-frequency class weights) into the checkpoint; eval doesn't
    # rebuild loss functions, so those keys are unused at inference time.
    model.load_state_dict(state["model"] if "model" in state else state, strict=False)
    model.eval()

    asp_pred_tags: list[list[str]] = []
    asp_gold_tags: list[list[str]] = []
    cau_pred_tags: list[list[str]] = []
    cau_gold_tags: list[list[str]] = []

    asp_pred_ent: set[tuple[int, int, int, str]] = set()
    asp_gold_ent: set[tuple[int, int, int, str]] = set()
    cau_pred_ent: set[tuple[int, int, int]] = set()
    cau_gold_ent: set[tuple[int, int, int]] = set()

    loader = DataLoader(ds, batch_size=8, shuffle=False)
    with torch.no_grad():
        b_offset = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"])
            asp_pred = out.asp_logits.argmax(-1).cpu().tolist()
            cau_pred = out.cause_logits.argmax(-1).cpu().tolist()
            asp_gold = batch["asp_labels"].cpu().tolist()
            cau_gold = batch["cause_labels"].cpu().tolist()

            for j in range(len(asp_pred)):
                ap = [ASPECT_SENT_ID2LABEL[p] for p, g in zip(asp_pred[j], asp_gold[j]) if g != IGNORE_INDEX]
                ag = [ASPECT_SENT_ID2LABEL[g] for g in asp_gold[j] if g != IGNORE_INDEX]
                cp = [CAUSE_ID2LABEL[p] for p, g in zip(cau_pred[j], cau_gold[j]) if g != IGNORE_INDEX]
                cg = [CAUSE_ID2LABEL[g] for g in cau_gold[j] if g != IGNORE_INDEX]
                asp_pred_tags.append(ap); asp_gold_tags.append(ag)
                cau_pred_tags.append(cp); cau_gold_tags.append(cg)

                doc_id = b_offset + j
                for s, e, base in _bio_segments(ap):
                    asp_pred_ent.add((doc_id, s, e, base))
                for s, e, base in _bio_segments(ag):
                    asp_gold_ent.add((doc_id, s, e, base))
                for s, e, _b in _bio_segments(cp):
                    cau_pred_ent.add((doc_id, s, e))
                for s, e, _b in _bio_segments(cg):
                    cau_gold_ent.add((doc_id, s, e))
            b_offset += len(asp_pred)

    print("=== aspect-sentiment (token-level, seqeval) ===")
    print(seqeval_report(asp_gold_tags, asp_pred_tags, digits=4))
    print("=== cause (token-level, seqeval) ===")
    print(seqeval_report(cau_gold_tags, cau_pred_tags, digits=4))

    p_a, r_a, f_a = _prf(asp_pred_ent, asp_gold_ent)
    p_c, r_c, f_c = _prf(cau_pred_ent, cau_gold_ent)
    print(f"=== entity-level F1 ===")
    print(f"aspect-sentiment: P={p_a:.4f} R={r_a:.4f} F1={f_a:.4f}")
    print(f"cause:            P={p_c:.4f} R={r_c:.4f} F1={f_c:.4f}")

    return {
        "asp_entity": {"p": p_a, "r": r_a, "f1": f_a},
        "cause_entity": {"p": p_c, "r": r_c, "f1": f_c},
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

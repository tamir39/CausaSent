"""PyTorch dataset for ABSA training (new schema).

Reads flat JSONL records produced by annotate_aspect_terms.py / annotate_tiki.py,
groups them by review id, and builds per-word label sequences for two heads:

  Head A (ATE): BIO + aspect_category, 15 labels — where the term is and what aspect
  Head B (Sentiment): binary, 2 labels — at B-token positions only (IGNORE_INDEX elsewhere)
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .label_schema import (
    ATE_LABEL2ID,
    IGNORE_INDEX,
    SENTIMENT_LABEL2ID,
    ate_tag,
)
from .segmenter import Segmenter, get_segmenter
from .span_align import expand_to_word_boundary, word_boundaries, words_in_span


@dataclass
class TaggedExample:
    review_id: str
    words: list[str]          # underscore-joined VnCoreNLP words
    ate_labels: list[int]     # per word, into ATE_LABEL2ID (Head A)
    sent_labels: list[int]    # per word, IGNORE_INDEX except at B-token positions (Head B)


def _build_word_labels(
    annotations: list[dict],
    words: list[str],
    text: str,
) -> tuple[list[int], list[int]]:
    """Convert annotation char-spans into per-word BIO labels for both heads."""
    n = len(words)
    w_spans = word_boundaries(text, words)
    if len(w_spans) != n:
        n = min(n, len(w_spans))

    ate = [ATE_LABEL2ID["O"]] * n
    sent = [IGNORE_INDEX] * n

    for ann in annotations:
        s_char, e_char = ann["aspect_term_span"]
        snapped = expand_to_word_boundary((s_char, e_char), w_spans)
        widx = words_in_span(snapped, w_spans)
        if not widx:
            continue
        asp = ann["aspect_category"]
        sentiment = ann["sentiment"]
        try:
            ate_tag(asp, "B")  # validate aspect_category is known
        except ValueError:
            continue  # unknown aspect_category — skip annotation
        if sentiment not in SENTIMENT_LABEL2ID:
            continue

        for k, wi in enumerate(widx):
            if wi >= n:
                continue
            pos = "B" if k == 0 else "I"
            tag = ate_tag(asp, pos)
            ate[wi] = ATE_LABEL2ID[tag]
            if k == 0:
                sent[wi] = SENTIMENT_LABEL2ID[sentiment]

    return ate, sent


def load_absa_jsonl(path: str | Path) -> list[dict]:
    """Load ABSA data into grouped reviews. Auto-detects format.

    Supports two formats:
      1. Review-level JSON array — `[{id, review, annotations: [...]}, ...]`
         (the canonical training format, written by build_ate_dataset.py)
      2. Annotation-level JSONL — one annotation per line; records sharing a
         review id are merged so downstream code sees `{id, review, annotations}`
         (the intermediate draft format from auto-annotation)
    """
    raw = Path(path).read_text(encoding="utf-8").lstrip("﻿").lstrip()
    if raw.startswith("["):
        return json.loads(raw)

    groups: dict[str, dict] = {}
    order: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = rec.get("id", "")
        if rid not in groups:
            groups[rid] = {"id": rid, "review": rec["review"], "annotations": []}
            order.append(rid)
        ann = {
            "aspect_term": rec.get("aspect_term", ""),
            "aspect_term_span": rec.get("aspect_term_span", [0, 0]),
            "aspect_category": rec.get("aspect_category", rec.get("aspect", "")),
            "sentiment": rec.get("sentiment", ""),
        }
        groups[rid]["annotations"].append(ann)
    return [groups[rid] for rid in order]


def build_tagged_examples(reviews: list[dict], segmenter: Segmenter) -> list[TaggedExample]:
    out: list[TaggedExample] = []
    for r in reviews:
        words = segmenter.segment(r["review"])
        if not words:
            continue
        ate, sent = _build_word_labels(r["annotations"], words, r["review"])
        out.append(TaggedExample(r["id"], words, ate, sent))
    return out


class TaggingDataset(Dataset):
    """Word-segmented reviews -> PhoBERT subword inputs with two label sequences.

    Per-word labels are propagated to the *first* subword of each word; subsequent
    subwords and special tokens are set to IGNORE_INDEX so they do not contribute
    to the loss.
    """

    def __init__(self, examples: list[TaggedExample], tokenizer, max_len: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        try:
            word_ids = enc.word_ids(batch_index=0)
        except (ValueError, AttributeError):
            word_ids = _word_ids_slow(self.tokenizer, ex.words, self.max_len)

        ate_labels = [IGNORE_INDEX] * len(word_ids)
        sent_labels = [IGNORE_INDEX] * len(word_ids)
        seen: set[int] = set()
        for i, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            if wid < len(ex.ate_labels):
                ate_labels[i] = ex.ate_labels[wid]
                sent_labels[i] = ex.sent_labels[wid]

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "ate_labels": torch.tensor(ate_labels, dtype=torch.long),
            "sent_labels": torch.tensor(sent_labels, dtype=torch.long),
        }


def _word_ids_slow(tokenizer, words: list[str], max_len: int) -> list[int | None]:
    """Manual word_ids() for slow tokenizers (PhoBERT often returns slow)."""
    cls = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    out: list[int | None] = [None]  # CLS
    n_special = int(cls is not None) + int(sep is not None)
    budget = max_len - n_special
    for w_idx, word in enumerate(words):
        toks = tokenizer.encode(word, add_special_tokens=False)
        if not toks:
            continue
        if len(out) - 1 + len(toks) > budget:
            break
        out.extend([w_idx] * len(toks))
    out.append(None)  # SEP
    while len(out) < max_len:
        out.append(None)
    return out[:max_len]


def load_tagging_dataset(
    path: str | Path,
    tokenizer,
    max_len: int,
    segmenter_kind: str = "vncorenlp",
) -> TaggingDataset:
    """Load JSONL data, segment, and build a TaggingDataset."""
    reviews = load_absa_jsonl(path)
    segmenter = get_segmenter(segmenter_kind)
    examples = build_tagged_examples(reviews, segmenter)
    return TaggingDataset(examples, tokenizer, max_len=max_len)

"""PyTorch datasets for the two training tasks.

`TaggingDataset` — for PhoBERT joint aspect-sentiment + cause extraction.
`ActionDataset`  — for mT5 action generation, conditioned on (aspect, sentiment, cause, review).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .label_schema import (
    ASPECT_SENT_LABEL2ID,
    CAUSE_LABEL2ID,
    IGNORE_INDEX,
    aspect_sent_tag,
)
from .schema import Review, load_reviews
from .segmenter import Segmenter, get_segmenter
from .span_align import expand_to_word_boundary, word_boundaries, words_in_span


@dataclass
class TaggedExample:
    review_id: str
    words: list[str]                 # underscore-joined VnCoreNLP words
    aspect_sent_labels: list[int]    # per word, into ASPECT_SENT_LABEL2ID
    cause_labels: list[int]          # per word, into CAUSE_LABEL2ID


def _build_word_labels(review: Review, words: list[str], text: str) -> tuple[list[int], list[int]]:
    """Convert annotation char-spans into per-word BIO labels for both heads."""
    n = len(words)
    asp_sent = [ASPECT_SENT_LABEL2ID["O"]] * n
    cause = [CAUSE_LABEL2ID["O"]] * n

    w_spans = word_boundaries(text, words)
    if len(w_spans) != n:
        # Misalignment — silently truncate to common length.
        n = min(n, len(w_spans))
        asp_sent = asp_sent[:n]
        cause = cause[:n]

    for ann in review.annotations:
        snapped = expand_to_word_boundary(tuple(ann.cause_span), w_spans)
        widx = words_in_span(snapped, w_spans)
        if not widx:
            continue
        for k, wi in enumerate(widx):
            pos = "B" if k == 0 else "I"
            tag = aspect_sent_tag(ann.aspect, ann.sentiment, pos)
            asp_sent[wi] = ASPECT_SENT_LABEL2ID[tag]
            cause[wi] = CAUSE_LABEL2ID[f"{pos}-CAUSE"]
    return asp_sent, cause


def build_tagged_examples(reviews: list[Review], segmenter: Segmenter) -> list[TaggedExample]:
    out: list[TaggedExample] = []
    for r in reviews:
        words = segmenter.segment(r.review)
        if not words:
            continue
        asp, cau = _build_word_labels(r, words, r.review)
        out.append(TaggedExample(r.id, words, asp, cau))
    return out


class TaggingDataset(Dataset):
    """Word-segmented reviews → PhoBERT subword inputs with two label sequences.

    Per-word labels are propagated to the *first* subword of each word; subsequent
    subwords and special tokens are set to IGNORE_INDEX so they do not contribute
    to the loss.
    """

    def __init__(
        self,
        examples: list[TaggedExample],
        tokenizer,
        max_len: int = 128,
    ):
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
        asp_labels = [IGNORE_INDEX] * len(word_ids)
        cau_labels = [IGNORE_INDEX] * len(word_ids)
        seen: set[int] = set()
        for i, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            if wid < len(ex.aspect_sent_labels):
                asp_labels[i] = ex.aspect_sent_labels[wid]
                cau_labels[i] = ex.cause_labels[wid]
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "asp_labels": torch.tensor(asp_labels, dtype=torch.long),
            "cause_labels": torch.tensor(cau_labels, dtype=torch.long),
        }


def _word_ids_slow(tokenizer, words: list[str], max_len: int) -> list[int | None]:
    """Manual word_ids() for slow tokenizers (PhoBERT often returns slow).

    Tokenizes each word individually and records which output position maps
    back to which word index. Mirrors what fast tokenizers expose via
    `enc.word_ids()`. Pads/truncates to `max_len` with None for specials/pad.
    """
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
        out.append(None)  # pad
    return out[:max_len]


def load_tagging_dataset(path: str | Path, tokenizer, max_len: int, segmenter_kind: str = "vncorenlp") -> TaggingDataset:
    reviews = load_reviews(path)
    segmenter = get_segmenter(segmenter_kind)
    examples = build_tagged_examples(reviews, segmenter)
    return TaggingDataset(examples, tokenizer, max_len=max_len)


# ---------------------------------------------------------------------------
# Action dataset (mT5)
# ---------------------------------------------------------------------------

ACTION_INPUT_TEMPLATE = (
    "aspect: {aspect}\nsentiment: {sentiment}\ncause: {cause}\nreview: {review}"
)


@dataclass
class ActionExample:
    input_text: str
    target_text: str


def build_action_examples(reviews: list[Review]) -> list[ActionExample]:
    out: list[ActionExample] = []
    for r in reviews:
        for ann in r.annotations:
            inp = ACTION_INPUT_TEMPLATE.format(
                aspect=ann.aspect,
                sentiment=ann.sentiment,
                cause=ann.cause_text,
                review=r.review,
            )
            out.append(ActionExample(inp, ann.action))
    return out


class ActionDataset(Dataset):
    def __init__(self, examples: list[ActionExample], tokenizer, max_input_len: int = 128, max_output_len: int = 32):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.input_text,
            truncation=True,
            max_length=self.max_input_len,
            padding="max_length",
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer() if hasattr(self.tokenizer, "as_target_tokenizer") else _noop():
            tgt = self.tokenizer(
                ex.target_text,
                truncation=True,
                max_length=self.max_output_len,
                padding="max_length",
                return_tensors="pt",
            )
        labels = tgt["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


class _noop:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def load_action_dataset(path: str | Path, tokenizer, max_input_len: int, max_output_len: int) -> ActionDataset:
    reviews = load_reviews(path)
    examples = build_action_examples(reviews)
    return ActionDataset(examples, tokenizer, max_input_len=max_input_len, max_output_len=max_output_len)

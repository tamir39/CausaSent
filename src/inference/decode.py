"""Decode PhoBERT two-head logits into structured (aspect, sentiment, cause) tuples."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..data.label_schema import (
    ASPECT_SENT_ID2LABEL,
    CAUSE_ID2LABEL,
    parse_aspect_sent_tag,
)


@dataclass
class ExtractedTuple:
    aspect: str
    sentiment: str
    cause_text: str
    cause_span: tuple[int, int]   # char span in original review
    word_indices: list[int]       # indices into the segmented word list
    confidence: float = 1.0       # mean predicted-class prob over the aspect span; 1.0 if probs unavailable


def _bio_segments(tags: list[str]) -> list[tuple[int, int, str]]:
    """Return list of (start_word_idx, end_word_idx_exclusive, base_tag) from BIO tags.

    `base_tag` is the part after the B-/I- prefix (e.g. 'DEL-NEG' or 'CAUSE').
    """
    segs: list[tuple[int, int, str]] = []
    i, n = 0, len(tags)
    while i < n:
        t = tags[i]
        if t == "O" or not t.startswith("B-"):
            i += 1
            continue
        base = t[2:]
        j = i + 1
        while j < n and tags[j] == f"I-{base}":
            j += 1
        segs.append((i, j, base))
        i = j
    return segs


def _span_confidence(
    asp_label_ids: Sequence[int],
    asp_probs: Sequence[Sequence[float]] | None,
    s: int,
    e: int,
) -> float:
    if asp_probs is None or e <= s:
        return 1.0
    vals: list[float] = []
    for k in range(s, e):
        if k >= len(asp_probs):
            break
        row = asp_probs[k]
        cls = asp_label_ids[k]
        if 0 <= cls < len(row):
            vals.append(float(row[cls]))
    return sum(vals) / len(vals) if vals else 1.0


def decode_to_tuples(
    asp_label_ids: list[int],
    cause_label_ids: list[int],
    word_spans: list[tuple[int, int]],
    review_text: str,
    *,
    asp_probs: Sequence[Sequence[float]] | None = None,
    min_confidence: float = 0.0,
) -> list[ExtractedTuple]:
    """Produce extracted tuples from per-word predictions.

    A tuple is created for every aspect-sentiment span; its cause is the
    cause-BIO span that overlaps it the most. If no cause overlaps, we fall
    back to the aspect-sentiment span itself.

    Post-processing:
    - drops empty-cause tuples
    - dedups identical (aspect, sentiment, cause_span)
    - filters by `min_confidence` (mean predicted-class prob over the aspect span)
    """
    asp_tags = [ASPECT_SENT_ID2LABEL[i] for i in asp_label_ids]
    cau_tags = [CAUSE_ID2LABEL[i] for i in cause_label_ids]

    asp_segs = _bio_segments(asp_tags)
    cau_segs = _bio_segments(cau_tags)

    out: list[ExtractedTuple] = []
    seen: set[tuple[str, str, int, int]] = set()
    for s, e, base in asp_segs:
        parsed = parse_aspect_sent_tag(f"B-{base}")
        if parsed is None:
            continue
        _, aspect, sentiment = parsed

        # Pick the cause segment with the largest overlap.
        best: tuple[int, int] | None = None
        best_ov = 0
        for cs, ce, _cb in cau_segs:
            ov = max(0, min(e, ce) - max(s, cs))
            if ov > best_ov:
                best_ov = ov
                best = (cs, ce)
        cs, ce = best if best is not None else (s, e)
        word_idx = list(range(cs, ce))
        if not word_idx:
            continue
        if word_idx[0] >= len(word_spans) or word_idx[-1] >= len(word_spans):
            continue
        char_s = word_spans[word_idx[0]][0]
        char_e = word_spans[word_idx[-1]][1]
        if char_e <= char_s:
            continue

        conf = _span_confidence(asp_label_ids, asp_probs, s, e)
        if conf < min_confidence:
            continue

        key = (aspect, sentiment, char_s, char_e)
        if key in seen:
            continue
        seen.add(key)

        out.append(ExtractedTuple(
            aspect=aspect,
            sentiment=sentiment,
            cause_text=review_text[char_s:char_e],
            cause_span=(char_s, char_e),
            word_indices=word_idx,
            confidence=conf,
        ))
    return out

"""Decode PhoBERT two-head logits into structured ABSA tuples.

Head A (ATE) BIO tags determine where aspect terms are and their categories.
Head B (Sentiment) logit at the B-token position determines sentiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..data.label_schema import (
    ATE_ID2LABEL,
    SENTIMENT_ID2LABEL,
    IGNORE_INDEX,
    parse_ate_tag,
)


@dataclass
class ExtractedTuple:
    aspect_category: str
    aspect_term: str                  # surface form extracted from review
    aspect_term_span: tuple[int, int] # char span in original review
    sentiment: str
    word_indices: list[int]           # word indices of the span
    confidence: float = 1.0           # mean B-token softmax probability


def _bio_segments(tags: list[str]) -> list[tuple[int, int, str]]:
    """Return (start_word_idx, end_word_idx_exclusive, base_tag) for each BIO span.

    `base_tag` is the part after B-/I- (e.g. 'delivery' or 'product_quality').
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


def decode_to_tuples(
    ate_label_ids: list[int],
    sent_label_ids: list[int],
    word_spans: list[tuple[int, int]],
    review_text: str,
    *,
    ate_probs: Sequence[Sequence[float]] | None = None,
    min_confidence: float = 0.0,
) -> list[ExtractedTuple]:
    """Produce extracted ABSA tuples from per-word predictions.

    For each B-token, the ATE head determines the span extent and aspect_category,
    and the Sentiment head gives the binary sentiment at that B-token position.

    Post-processing:
    - Dedups identical (aspect_category, sentiment, char_span) tuples
    - Filters by min_confidence
    """
    ate_tags = [ATE_ID2LABEL.get(i, "O") for i in ate_label_ids]

    out: list[ExtractedTuple] = []
    seen: set[tuple[str, str, int, int]] = set()

    for s, e, base in _bio_segments(ate_tags):
        parsed = parse_ate_tag(f"B-{base}")
        if parsed is None:
            continue
        _, aspect_category = parsed

        # Sentiment from Head B at the B-token position (index s).
        s_lbl = sent_label_ids[s] if s < len(sent_label_ids) else -1
        if s_lbl == IGNORE_INDEX or s_lbl < 0 or s_lbl >= len(SENTIMENT_ID2LABEL):
            # Fall back to positive when no label available (inference without labels).
            sentiment = SENTIMENT_ID2LABEL.get(0, "positive")
        else:
            sentiment = SENTIMENT_ID2LABEL[s_lbl]

        # Map word span → char span.
        word_idx = list(range(s, e))
        if not word_idx:
            continue
        if word_idx[0] >= len(word_spans) or word_idx[-1] >= len(word_spans):
            continue
        char_s = word_spans[word_idx[0]][0]
        char_e = word_spans[word_idx[-1]][1]
        if char_e <= char_s:
            continue

        # Confidence: mean ATE softmax prob over the span.
        conf = 1.0
        if ate_probs is not None:
            vals = [
                float(ate_probs[k][ate_label_ids[k]])
                for k in range(s, e)
                if k < len(ate_probs) and 0 <= ate_label_ids[k] < len(ate_probs[k])
            ]
            conf = sum(vals) / len(vals) if vals else 1.0

        if conf < min_confidence:
            continue

        key = (aspect_category, sentiment, char_s, char_e)
        if key in seen:
            continue
        seen.add(key)

        out.append(ExtractedTuple(
            aspect_category=aspect_category,
            aspect_term=review_text[char_s:char_e],
            aspect_term_span=(char_s, char_e),
            sentiment=sentiment,
            word_indices=word_idx,
            confidence=conf,
        ))

    return out

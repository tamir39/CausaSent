"""Decode PhoBERT two-head logits into structured (aspect, sentiment, cause) tuples."""
from __future__ import annotations

from dataclasses import dataclass

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


def decode_to_tuples(
    asp_label_ids: list[int],
    cause_label_ids: list[int],
    word_spans: list[tuple[int, int]],
    review_text: str,
) -> list[ExtractedTuple]:
    """Produce extracted tuples from per-word predictions.

    A tuple is created for every aspect-sentiment span; its cause is the
    cause-BIO span that overlaps it the most. If no cause overlaps, we fall
    back to the aspect-sentiment span itself.
    """
    asp_tags = [ASPECT_SENT_ID2LABEL[i] for i in asp_label_ids]
    cau_tags = [CAUSE_ID2LABEL[i] for i in cause_label_ids]

    asp_segs = _bio_segments(asp_tags)
    cau_segs = _bio_segments(cau_tags)

    out: list[ExtractedTuple] = []
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
        char_s = word_spans[word_idx[0]][0]
        char_e = word_spans[word_idx[-1]][1]
        out.append(ExtractedTuple(
            aspect=aspect,
            sentiment=sentiment,
            cause_text=review_text[char_s:char_e],
            cause_span=(char_s, char_e),
            word_indices=word_idx,
        ))
    return out

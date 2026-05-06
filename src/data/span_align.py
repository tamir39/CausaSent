"""Char-span helpers: locate a cause substring and snap to word boundaries.

Word boundaries here mean *VnCoreNLP word boundaries* once the review has been
word-segmented. We treat any span that does not align with those boundaries as
a misalignment and expand outward to the nearest enclosing word edges.
"""
from __future__ import annotations

import re
import unicodedata


def _normalize(s: str) -> str:
    """NFC-normalize and lowercase for fuzzy substring matching."""
    return unicodedata.normalize("NFC", s).lower()


def find_cause_span(review: str, cause_text: str) -> tuple[int, int] | None:
    """Locate `cause_text` in `review`. Tries exact then case-insensitive match.

    Returns [start, end) char span, or None if not found.
    """
    if not cause_text:
        return None
    idx = review.find(cause_text)
    if idx >= 0:
        return idx, idx + len(cause_text)

    rev_n = _normalize(review)
    cau_n = _normalize(cause_text)
    idx = rev_n.find(cau_n)
    if idx >= 0:
        return idx, idx + len(cau_n)
    return None


def word_boundaries(text: str, segmented_words: list[str]) -> list[tuple[int, int]]:
    """Map each segmented word back to its [start, end) char span in the original `text`.

    `segmented_words` is the output of VnCoreNLP word segmentation, where each entry
    may use underscores to join syllables (e.g. "giao_hàng"). We strip underscores
    before searching so spans align to the original review text.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for w in segmented_words:
        surface = w.replace("_", " ")
        # Skip leading whitespace in the source.
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        idx = text.find(surface, cursor)
        if idx < 0:
            # Fallback: regex match collapsing internal whitespace differences.
            pat = re.escape(surface).replace(r"\ ", r"\s+")
            m = re.search(pat, text[cursor:])
            if not m:
                continue
            idx = cursor + m.start()
            end = cursor + m.end()
        else:
            end = idx + len(surface)
        spans.append((idx, end))
        cursor = end
    return spans


def expand_to_word_boundary(
    span: tuple[int, int], word_spans: list[tuple[int, int]]
) -> tuple[int, int]:
    """Snap a [start, end) char span outward to the enclosing word boundaries.

    If the span starts mid-word, move start back to that word's start.
    If the span ends mid-word, move end forward to that word's end.
    Words fully inside [start, end) are preserved.
    """
    s, e = span
    new_s, new_e = s, e
    for ws, we in word_spans:
        if ws <= s < we:
            new_s = min(new_s, ws)
        if ws < e <= we:
            new_e = max(new_e, we)
    return new_s, new_e


def words_in_span(
    span: tuple[int, int], word_spans: list[tuple[int, int]]
) -> list[int]:
    """Indices of `word_spans` whose char range overlaps `span`."""
    s, e = span
    return [i for i, (ws, we) in enumerate(word_spans) if ws < e and we > s]

"""Span alignment helpers."""
from src.data.span_align import (
    expand_to_word_boundary,
    find_cause_span,
    word_boundaries,
    words_in_span,
)


def test_find_cause_span_exact():
    assert find_cause_span("Ship lâu nhưng đóng gói đẹp", "Ship lâu") == (0, 8)


def test_find_cause_span_case_insensitive():
    # The model wrote "ship lâu" lowercase but the review starts with "Ship".
    out = find_cause_span("Ship lâu nhưng đẹp", "ship lâu")
    assert out == (0, 8)


def test_find_cause_span_no_match():
    assert find_cause_span("Ship lâu", "tốc độ") is None


def test_find_cause_span_empty_returns_none():
    assert find_cause_span("Ship lâu", "") is None


def test_word_boundaries_basic():
    text = "Ship lâu nhưng đóng gói đẹp"
    words = ["Ship", "lâu", "nhưng", "đóng_gói", "đẹp"]
    spans = word_boundaries(text, words)
    # "đóng_gói" should map to chars covering "đóng gói".
    assert spans[0] == (0, 4)
    assert spans[1] == (5, 8)
    assert spans[2] == (9, 14)
    assert text[spans[3][0]:spans[3][1]] == "đóng gói"
    assert text[spans[4][0]:spans[4][1]] == "đẹp"


def test_expand_to_word_boundary_starts_midword():
    # Word boundaries: [0,4)="Ship", [5,8)="lâu", [9,14)="nhưng"
    word_spans = [(0, 4), (5, 8), (9, 14)]
    # Span (2, 7) starts mid "Ship" and ends mid "lâu" → should expand to (0, 8).
    assert expand_to_word_boundary((2, 7), word_spans) == (0, 8)


def test_expand_to_word_boundary_already_aligned():
    word_spans = [(0, 4), (5, 8)]
    assert expand_to_word_boundary((0, 8), word_spans) == (0, 8)


def test_expand_to_word_boundary_no_change_when_outside_words():
    word_spans = [(0, 4), (5, 8)]
    # Span entirely between words — no expansion.
    assert expand_to_word_boundary((4, 5), word_spans) == (4, 5)


def test_words_in_span():
    word_spans = [(0, 4), (5, 8), (9, 14), (15, 23)]
    # Span covering words 1 and 2 only.
    assert words_in_span((5, 14), word_spans) == [1, 2]
    # Span overlapping word 0 partially.
    assert words_in_span((2, 6), word_spans) == [0, 1]
    # Span covering nothing.
    assert words_in_span((4, 5), word_spans) == []

"""BIO decoding → ExtractedTuple."""
from src.data.label_schema import (
    ASPECT_SENT_LABEL2ID,
    CAUSE_LABEL2ID,
)
from src.inference.decode import _bio_segments, decode_to_tuples


def test_bio_segments_empty():
    assert _bio_segments([]) == []
    assert _bio_segments(["O", "O", "O"]) == []


def test_bio_segments_single():
    tags = ["B-DEL-NEG"]
    assert _bio_segments(tags) == [(0, 1, "DEL-NEG")]


def test_bio_segments_b_i_i():
    tags = ["B-DEL-NEG", "I-DEL-NEG", "I-DEL-NEG"]
    assert _bio_segments(tags) == [(0, 3, "DEL-NEG")]


def test_bio_segments_multiple_disjoint():
    tags = ["B-DEL-NEG", "I-DEL-NEG", "O", "B-PACK-POS"]
    assert _bio_segments(tags) == [(0, 2, "DEL-NEG"), (3, 4, "PACK-POS")]


def test_bio_segments_b_followed_by_different_i_starts_new():
    # An I- with a different base does NOT continue the previous segment.
    tags = ["B-DEL-NEG", "I-PACK-POS"]
    out = _bio_segments(tags)
    # First is a 1-token segment; the stray I- is ignored (we walk forward, no B→ skip).
    assert out == [(0, 1, "DEL-NEG")]


def _label_seq(asp_tags: list[str]) -> list[int]:
    return [ASPECT_SENT_LABEL2ID[t] for t in asp_tags]


def _cause_seq(cause_tags: list[str]) -> list[int]:
    return [CAUSE_LABEL2ID[t] for t in cause_tags]


def test_decode_to_tuples_basic():
    review = "Ship lâu nhưng đóng gói đẹp"
    word_spans = [(0, 4), (5, 8), (9, 14), (15, 23), (24, 27)]
    asp_tags = ["B-DEL-NEG", "I-DEL-NEG", "O", "B-PACK-POS", "I-PACK-POS"]
    cau_tags = ["B-CAUSE", "I-CAUSE", "O", "B-CAUSE", "I-CAUSE"]

    tuples = decode_to_tuples(_label_seq(asp_tags), _cause_seq(cau_tags), word_spans, review)
    assert len(tuples) == 2

    t0, t1 = tuples
    assert (t0.aspect, t0.sentiment) == ("delivery", "negative")
    assert t0.cause_text == "Ship lâu"
    assert t0.cause_span == (0, 8)

    assert (t1.aspect, t1.sentiment) == ("packaging", "positive")
    assert t1.cause_text == "đóng gói đẹp"
    assert t1.cause_span == (15, 27)


def test_decode_dedups_identical_tuples():
    # Two B-DEL-NEG segments back-to-back at the same char span shouldn't appear twice.
    # We can't actually produce dup spans from BIO alone, but we can simulate the post-dedup
    # contract by feeding two adjacent B-* segments that map to overlapping cause overlap
    # logic — here we verify dedup key uses (aspect, sentiment, char_s, char_e).
    review = "Ship lâu nhé"
    word_spans = [(0, 4), (5, 8), (9, 12)]
    asp_tags = ["B-DEL-NEG", "B-DEL-NEG", "O"]
    cau_tags = ["O", "O", "O"]
    tuples = decode_to_tuples(_label_seq(asp_tags), _cause_seq(cau_tags), word_spans, review)
    # Two distinct word spans → two distinct char spans → no dedup.
    assert len(tuples) == 2
    # All confidences default to 1.0 when probs not supplied.
    assert all(t.confidence == 1.0 for t in tuples)


def test_decode_min_confidence_filters():
    review = "Ship lâu nhé"
    word_spans = [(0, 4), (5, 8), (9, 12)]
    asp_tags = ["B-DEL-NEG", "I-DEL-NEG", "O"]
    cau_tags = ["O", "O", "O"]
    asp_ids = _label_seq(asp_tags)
    # Build a probs matrix where the predicted-class prob is 0.4 — below threshold 0.6.
    n_classes = max(asp_ids) + 1
    probs = []
    for cls in asp_ids:
        row = [0.0] * (n_classes + 5)
        row[cls] = 0.4
        probs.append(row)
    tuples = decode_to_tuples(
        asp_ids, _cause_seq(cau_tags), word_spans, review,
        asp_probs=probs, min_confidence=0.6,
    )
    assert tuples == []


def test_decode_falls_back_when_no_cause_overlap():
    # An aspect span with no cause overlap → cause defaults to the aspect span itself.
    review = "Giá đắt"
    word_spans = [(0, 3), (4, 7)]
    asp_tags = ["B-PRICE-NEG", "I-PRICE-NEG"]
    cau_tags = ["O", "O"]

    tuples = decode_to_tuples(_label_seq(asp_tags), _cause_seq(cau_tags), word_spans, review)
    assert len(tuples) == 1
    t = tuples[0]
    assert t.aspect == "price"
    assert t.sentiment == "negative"
    assert t.cause_span == (0, 7)
    assert t.cause_text == "Giá đắt"

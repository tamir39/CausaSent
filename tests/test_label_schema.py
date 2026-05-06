"""Sanity checks on the BIO label spaces."""
from src.data.label_schema import (
    ASPECT_SENT_LABELS,
    ASPECT_SENT_LABEL2ID,
    CAUSE_LABELS,
    CAUSE_LABEL2ID,
    aspect_sent_tag,
    parse_aspect_sent_tag,
)


def test_aspect_sent_label_count():
    # 1 (O) + 7 aspects * 3 sentiments * 2 (B/I) = 43
    assert len(ASPECT_SENT_LABELS) == 43
    assert ASPECT_SENT_LABELS[0] == "O"
    assert len(set(ASPECT_SENT_LABELS)) == 43  # unique


def test_aspect_sent_label2id_inverse():
    for tag, idx in ASPECT_SENT_LABEL2ID.items():
        assert ASPECT_SENT_LABELS[idx] == tag


def test_cause_label_count():
    assert CAUSE_LABELS == ["O", "B-CAUSE", "I-CAUSE"]
    assert CAUSE_LABEL2ID["B-CAUSE"] == 1
    assert CAUSE_LABEL2ID["I-CAUSE"] == 2


def test_aspect_sent_tag_roundtrip():
    tag = aspect_sent_tag("delivery", "negative", "B")
    assert tag == "B-DEL-NEG"
    parsed = parse_aspect_sent_tag(tag)
    assert parsed == ("B", "delivery", "negative")


def test_aspect_sent_tag_all_combinations_resolve():
    from src.data.label_schema import ASPECTS, SENTIMENTS
    for asp in ASPECTS:
        for sent in SENTIMENTS:
            for pos in ("B", "I"):
                tag = aspect_sent_tag(asp, sent, pos)
                assert tag in ASPECT_SENT_LABEL2ID
                assert parse_aspect_sent_tag(tag) == (pos, asp, sent)


def test_parse_O_returns_none():
    assert parse_aspect_sent_tag("O") is None


def test_aspect_sent_tag_rejects_bad_position():
    try:
        aspect_sent_tag("delivery", "negative", "X")
    except ValueError:
        return
    raise AssertionError("expected ValueError for bad position")

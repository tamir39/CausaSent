"""Validator tests — pure-function pipeline, no LLM."""
from src.data.validate import (
    ValidationStats,
    dedup_and_resolve_overlaps,
    validate_annotation,
    validate_review,
)


REVIEW = "Ship lâu nhưng đóng gói đẹp"


def _ann(**kw):
    base = {
        "aspect": "delivery",
        "sentiment": "negative",
        "cause": "Ship lâu",
        "action": "Rút ngắn thời gian giao",
    }
    base.update(kw)
    return base


def test_valid_passthrough():
    s = ValidationStats()
    ann = validate_annotation(_ann(), REVIEW, stats=s)
    assert ann is not None
    assert ann.cause_span == (0, 8)
    assert ann.cause_text == "Ship lâu"
    assert s.kept == 0  # validate_annotation does not bump kept; that's validate_review's job


def test_bad_aspect_dropped():
    s = ValidationStats()
    assert validate_annotation(_ann(aspect="warranty"), REVIEW, stats=s) is None
    assert s.dropped_bad_aspect == 1


def test_bad_sentiment_dropped():
    s = ValidationStats()
    assert validate_annotation(_ann(sentiment="meh"), REVIEW, stats=s) is None
    assert s.dropped_bad_sentiment == 1


def test_hallucinated_cause_dropped():
    s = ValidationStats()
    # "tốc độ" is not a substring of REVIEW.
    assert validate_annotation(_ann(cause="tốc độ"), REVIEW, stats=s) is None
    assert s.dropped_span_not_found == 1


def test_empty_action_dropped():
    s = ValidationStats()
    assert validate_annotation(_ann(action="   "), REVIEW, stats=s) is None
    assert s.dropped_empty_action == 1


def test_action_too_long_dropped():
    s = ValidationStats()
    long_action = " ".join(["xin"] * 11)
    assert validate_annotation(_ann(action=long_action), REVIEW, stats=s) is None
    assert s.dropped_action_too_long == 1


def test_dedup_and_overlap():
    s = ValidationStats()
    review = validate_review(
        "r1",
        REVIEW,
        [
            _ann(),                                           # delivery / "Ship lâu"
            _ann(),                                           # exact dup → dropped_duplicate
            _ann(aspect="packaging", sentiment="positive",
                 cause="đóng gói đẹp",
                 action="Duy trì cách đóng gói"),             # different aspect, disjoint span
            _ann(aspect="product_quality", sentiment="negative",
                 cause="lâu nhưng",
                 action="Kiểm tra"),                          # overlaps "Ship lâu" → dropped_overlap
        ],
        stats=s,
    )
    assert review is not None
    assert len(review.annotations) == 2
    assert s.dropped_duplicate == 1
    assert s.dropped_overlap == 1
    assert s.kept == 2


def test_explicit_span_matching_text_used():
    s = ValidationStats()
    raw = _ann(cause_span=[0, 8])
    ann = validate_annotation(raw, REVIEW, stats=s)
    assert ann is not None
    assert ann.cause_span == (0, 8)


def test_explicit_span_mismatch_falls_back_to_search():
    s = ValidationStats()
    raw = _ann(cause_span=[100, 200])  # bogus span
    ann = validate_annotation(raw, REVIEW, stats=s)
    # Falls back to substring search and succeeds.
    assert ann is not None
    assert ann.cause_span == (0, 8)

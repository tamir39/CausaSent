"""Pydantic schema validation: span/text consistency, taxonomy enforcement."""
from pydantic import ValidationError

from src.data.schema import Annotation, Review, load_reviews


def _good_review() -> dict:
    return {
        "id": "t1",
        "review": "Ship lâu nhưng đóng gói đẹp",
        "annotations": [
            {
                "aspect": "delivery",
                "sentiment": "negative",
                "cause_text": "Ship lâu",
                "cause_span": [0, 8],
                "action": "Cải thiện giao hàng",
            }
        ],
    }


def test_valid_review_passes():
    Review.model_validate(_good_review())


def test_bad_aspect_rejected():
    payload = _good_review()
    payload["annotations"][0]["aspect"] = "vibes"
    try:
        Review.model_validate(payload)
    except ValidationError:
        return
    raise AssertionError("expected validation error for bad aspect")


def test_bad_sentiment_rejected():
    payload = _good_review()
    payload["annotations"][0]["sentiment"] = "happy"
    try:
        Review.model_validate(payload)
    except ValidationError:
        return
    raise AssertionError("expected validation error for bad sentiment")


def test_span_text_mismatch_rejected():
    payload = _good_review()
    payload["annotations"][0]["cause_span"] = [0, 4]   # "Ship", but cause_text is "Ship lâu"
    try:
        Review.model_validate(payload)
    except ValidationError:
        return
    raise AssertionError("expected validation error for span/text mismatch")


def test_span_out_of_range_rejected():
    payload = _good_review()
    payload["annotations"][0]["cause_span"] = [0, 999]
    try:
        Review.model_validate(payload)
    except ValidationError:
        return
    raise AssertionError("expected validation error for out-of-range span")


def test_malformed_span_rejected():
    try:
        Annotation(
            aspect="delivery",
            sentiment="negative",
            cause_text="x",
            cause_span=(5, 5),  # start == end
            action="do x",
        )
    except ValidationError:
        return
    raise AssertionError("expected validation error for malformed span")


def test_load_reviews_sample():
    reviews = load_reviews("scripts/sample_data.json")
    assert len(reviews) == 2
    assert reviews[0].id == "s001"
    assert len(reviews[0].annotations) == 2
    # Spans are exact substrings.
    for r in reviews:
        for ann in r.annotations:
            s, e = ann.cause_span
            assert r.review[s:e] == ann.cause_text

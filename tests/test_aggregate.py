"""Tests for src.inference.aggregate.

Synthesises PredictedTuple objects without touching the model, so these are
fast and CPU-only.
"""
from __future__ import annotations

from src.inference.aggregate import (
    aggregate,
    normalize_term,
    rank_by_priority,
)
from src.inference.pipeline import PredictedTuple


def _t(asp: str, sent: str, term: str, conf: float = 0.95) -> PredictedTuple:
    return PredictedTuple(
        aspect_category=asp,
        aspect_term=term,
        aspect_term_span=(0, len(term)),
        sentiment=sent,
        confidence=conf,
    )


def test_normalize_term_folds_punct_and_case():
    assert normalize_term("Hàng") == "hàng"
    assert normalize_term("Hàng.") == "hàng"
    assert normalize_term("  Hàng,  ") == "hàng"
    assert normalize_term("HÀNG đẹp!") == "hàng đẹp"


def test_aggregate_basic_counts():
    reviews = [
        [_t("delivery", "negative", "Hàng"), _t("packaging", "negative", "hộp")],
        [_t("delivery", "negative", "hàng")],
        [_t("price", "positive", "giá")],
        [],
    ]
    s = aggregate(reviews, top_k=5)
    assert s.n_reviews == 4
    assert s.n_reviews_with_tuples == 3
    assert s.n_tuples == 4
    assert "delivery" in s.aspects
    assert s.aspects["delivery"].negative == 2
    assert s.aspects["delivery"].positive == 0
    assert s.aspects["delivery"].negative_ratio == 1.0
    # The two "Hàng" / "hàng" should fold into one normalized term.
    top = dict(s.aspects["delivery"].cells["negative"].top_terms)
    assert top.get("hàng") == 2


def test_aggregate_drops_low_confidence():
    reviews = [
        [_t("delivery", "negative", "Hàng", conf=0.4)],
        [_t("delivery", "negative", "hàng", conf=0.95)],
    ]
    s = aggregate(reviews, top_k=5, min_confidence=0.5)
    assert s.aspects["delivery"].negative == 1


def test_aggregate_rejects_out_of_taxonomy():
    reviews = [
        [_t("wifi", "negative", "wifi")],  # not in ASPECTS
        [_t("delivery", "neutral", "ship")],  # not in SENTIMENTS
        [_t("delivery", "negative", "ship")],
    ]
    s = aggregate(reviews)
    assert s.n_tuples == 1
    assert s.aspects["delivery"].negative == 1
    assert "wifi" not in s.aspects


def test_rank_by_priority_orders_high_negative_first():
    # delivery: 5 neg / 0 pos → ratio 1.0, score 5
    # packaging: 2 neg / 8 pos → ratio 0.2, score 0.4
    reviews = (
        [[_t("delivery", "negative", "ship")] for _ in range(5)]
        + [[_t("packaging", "negative", "hộp")] for _ in range(2)]
        + [[_t("packaging", "positive", "đẹp")] for _ in range(8)]
    )
    s = aggregate(reviews)
    ordered = rank_by_priority(s)
    assert ordered[0].aspect_category == "delivery"


def test_top_terms_limited_by_k():
    reviews = [
        [_t("delivery", "negative", f"term{i}")] for i in range(10)
    ]
    s = aggregate(reviews, top_k=3)
    assert len(s.aspects["delivery"].cells["negative"].top_terms) == 3

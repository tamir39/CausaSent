"""Tests for src.inference.action_llm.

We pin GEMINI_API_KEY to '' so generate_actions() falls back to templates
deterministically. No network calls.
"""
from __future__ import annotations

import os

from src.inference.action_llm import (
    _parse_response,
    generate_actions,
    template_actions,
)
from src.inference.aggregate import aggregate
from src.inference.pipeline import PredictedTuple


def _t(asp: str, sent: str, term: str) -> PredictedTuple:
    return PredictedTuple(
        aspect_category=asp,
        aspect_term=term,
        aspect_term_span=(0, len(term)),
        sentiment=sent,
        confidence=0.95,
    )


def _summary_with_two_aspects():
    return aggregate(
        [
            [_t("delivery", "negative", "ship chậm")],
            [_t("delivery", "negative", "giao chậm")],
            [_t("delivery", "negative", "ship chậm")],
            [_t("delivery", "negative", "ship chậm")],
            [_t("delivery", "negative", "ship chậm")],
            [_t("packaging", "positive", "đóng gói đẹp")],
            [_t("packaging", "positive", "đóng gói đẹp")],
        ]
    )


def test_template_actions_covers_every_cell():
    s = _summary_with_two_aspects()
    actions = template_actions(s)
    cells = {(a.aspect_category, a.sentiment) for a in actions}
    assert ("delivery", "negative") in cells
    assert ("packaging", "positive") in cells


def test_template_priority_high_for_strong_negatives():
    s = _summary_with_two_aspects()
    actions = template_actions(s)
    by_key = {(a.aspect_category, a.sentiment): a for a in actions}
    # 5 neg / 0 pos → ratio 1.0, count 5 → "high"
    assert by_key[("delivery", "negative")].priority == "high"


def test_template_includes_evidence_terms():
    s = _summary_with_two_aspects()
    actions = template_actions(s)
    delivery_neg = next(
        a for a in actions if a.aspect_category == "delivery" and a.sentiment == "negative"
    )
    assert "ship chậm" in delivery_neg.evidence_terms


def test_generate_actions_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s = _summary_with_two_aspects()
    actions = generate_actions(s)
    assert actions  # not empty
    # Template-fallback strings are deterministic and contain Vietnamese text.
    delivery_neg = next(
        a for a in actions if a.aspect_category == "delivery" and a.sentiment == "negative"
    )
    assert "ship" in delivery_neg.action.lower() or "giao" in delivery_neg.action.lower()


def test_generate_actions_empty_summary():
    empty = aggregate([])
    monkeypatched_key = os.environ.pop("GEMINI_API_KEY", None)
    try:
        actions = generate_actions(empty)
    finally:
        if monkeypatched_key is not None:
            os.environ["GEMINI_API_KEY"] = monkeypatched_key
    assert actions == []


def test_parse_response_strips_markdown_fence():
    raw = """```json
[{"aspect_category":"delivery","sentiment":"negative","action":"x","priority":"high","evidence_terms":[]}]
```"""
    parsed = _parse_response(raw)
    assert isinstance(parsed, list)
    assert parsed[0]["aspect_category"] == "delivery"


def test_parse_response_plain_json():
    parsed = _parse_response('[{"aspect_category":"x","sentiment":"positive","action":"a","priority":"low","evidence_terms":[]}]')
    assert parsed[0]["sentiment"] == "positive"

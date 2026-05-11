"""Closed-set aspect/sentiment taxonomy and label spaces for the new ABSA model.

Head A (ATE): BIO span tagger with aspect_category encoded in the tag.
  Labels: O, B-delivery, I-delivery, ..., B-appearance, I-appearance  (15 total)

Head B (Sentiment): binary classifier applied at B-token positions only.
  Labels: positive (0), negative (1)  (2 total)
"""
from __future__ import annotations

# Canonical aspect names (closed set — never extend without a PRD update).
ASPECTS: tuple[str, ...] = (
    "delivery",
    "packaging",
    "product_quality",
    "price",
    "customer_service",
    "usability",
    "appearance",
)

ASPECT_CODES: dict[str, str] = {
    "delivery": "DEL",
    "packaging": "PACK",
    "product_quality": "QUAL",
    "price": "PRICE",
    "customer_service": "SVC",
    "usability": "USE",
    "appearance": "APP",
}
CODE_TO_ASPECT: dict[str, str] = {v: k for k, v in ASPECT_CODES.items()}

# Binary sentiments only (neutral dropped — only 2.7% of gold data).
SENTIMENTS: tuple[str, ...] = ("positive", "negative")

# Head A: ATE + aspect_category BIO (1 + 7*2 = 15 labels).
def _build_ate_labels() -> list[str]:
    labels = ["O"]
    for asp in ASPECTS:
        labels.append(f"B-{asp}")
        labels.append(f"I-{asp}")
    return labels


ATE_LABELS: list[str] = _build_ate_labels()
ATE_LABEL2ID: dict[str, int] = {l: i for i, l in enumerate(ATE_LABELS)}
ATE_ID2LABEL: dict[int, str] = {i: l for l, i in ATE_LABEL2ID.items()}

# Head B: binary sentiment (2 labels, only computed at B-token positions).
SENTIMENT_LABELS: list[str] = ["positive", "negative"]
SENTIMENT_LABEL2ID: dict[str, int] = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
SENTIMENT_ID2LABEL: dict[int, str] = {i: l for l, i in SENTIMENT_LABEL2ID.items()}

# Sentinel for sub-word / pad / special tokens — ignored by CrossEntropy.
IGNORE_INDEX: int = -100


def ate_tag(aspect_category: str, position: str) -> str:
    """Build an ATE BIO tag like 'B-delivery'. `position` in {'B', 'I'}."""
    if position not in {"B", "I"}:
        raise ValueError(f"position must be B or I, got {position!r}")
    if aspect_category not in ASPECTS:
        raise ValueError(f"aspect_category {aspect_category!r} not in taxonomy")
    return f"{position}-{aspect_category}"


def parse_ate_tag(tag: str) -> tuple[str, str] | None:
    """Inverse of `ate_tag`. Returns (position, aspect_category) or None for 'O'."""
    if tag == "O":
        return None
    parts = tag.split("-", 1)
    if len(parts) != 2:
        return None
    pos, asp = parts
    if pos not in {"B", "I"} or asp not in ASPECTS:
        return None
    return pos, asp

"""Closed-set aspect/sentiment taxonomy and BIO label spaces."""
from __future__ import annotations

# Canonical aspect names (from SPEC) and their short codes used in BIO tags.
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

SENTIMENTS: tuple[str, ...] = ("positive", "negative", "neutral")
SENTIMENT_CODES: dict[str, str] = {"positive": "POS", "negative": "NEG", "neutral": "NEU"}
CODE_TO_SENTIMENT: dict[str, str] = {v: k for k, v in SENTIMENT_CODES.items()}


def _build_aspect_sentiment_labels() -> list[str]:
    labels = ["O"]
    for asp in ASPECTS:
        ac = ASPECT_CODES[asp]
        for sent in SENTIMENTS:
            sc = SENTIMENT_CODES[sent]
            labels.append(f"B-{ac}-{sc}")
            labels.append(f"I-{ac}-{sc}")
    return labels


# Head A: aspect-sentiment BIO (1 + 7 * 3 * 2 = 43 labels).
ASPECT_SENT_LABELS: list[str] = _build_aspect_sentiment_labels()
ASPECT_SENT_LABEL2ID: dict[str, int] = {l: i for i, l in enumerate(ASPECT_SENT_LABELS)}
ASPECT_SENT_ID2LABEL: dict[int, str] = {i: l for l, i in ASPECT_SENT_LABEL2ID.items()}

# Head B: cause BIO (3 labels).
CAUSE_LABELS: list[str] = ["O", "B-CAUSE", "I-CAUSE"]
CAUSE_LABEL2ID: dict[str, int] = {l: i for i, l in enumerate(CAUSE_LABELS)}
CAUSE_ID2LABEL: dict[int, str] = {i: l for l, i in CAUSE_LABEL2ID.items()}

# Sentinel for sub-word / pad / special tokens — ignored by CrossEntropy.
IGNORE_INDEX: int = -100


def aspect_sent_tag(aspect: str, sentiment: str, position: str) -> str:
    """Build a BIO tag like 'B-DEL-NEG'. `position` ∈ {'B', 'I'}."""
    if position not in {"B", "I"}:
        raise ValueError(f"position must be B or I, got {position!r}")
    return f"{position}-{ASPECT_CODES[aspect]}-{SENTIMENT_CODES[sentiment]}"


def parse_aspect_sent_tag(tag: str) -> tuple[str, str, str] | None:
    """Inverse of `aspect_sent_tag`. Returns (position, aspect, sentiment) or None for 'O'."""
    if tag == "O":
        return None
    pos, ac, sc = tag.split("-")
    return pos, CODE_TO_ASPECT[ac], CODE_TO_SENTIMENT[sc]

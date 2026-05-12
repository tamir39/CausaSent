"""Aggregate per-review ABSA tuples into a business-level summary.

Given N reviews each producing M tuples `(aspect_category, aspect_term, sentiment, ...)`,
this module collapses them into one frequency-ranked picture per aspect_category:

  - sentiment distribution per aspect (counts + ratio of negatives)
  - top-K aspect_terms per (aspect_category, sentiment) cell
  - overall coverage stats (how many reviews mention each aspect)

Output is plain dataclasses / dicts so the downstream action LLM and the PWA
can both consume it without an extra serialisation hop.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..data.label_schema import ASPECTS, SENTIMENTS
from .pipeline import PredictedTuple


# ---- Normalisation -----------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_term(term: str) -> str:
    """Lowercase + NFC + strip punctuation + collapse spaces.

    Used as the grouping key when we count `aspect_term` occurrences so that
    "Hàng", "hàng", "hàng." all fold together.
    """
    t = unicodedata.normalize("NFC", term).strip().lower()
    t = _PUNCT_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


# ---- Result types ------------------------------------------------------------

@dataclass
class AspectCell:
    """One cell of the (aspect_category × sentiment) summary grid."""
    aspect_category: str
    sentiment: str
    count: int
    top_terms: list[tuple[str, int]] = field(default_factory=list)  # [(term, freq)]


@dataclass
class AspectSummary:
    """All sentiment cells for a single aspect_category, plus rollups."""
    aspect_category: str
    total_mentions: int
    n_reviews: int            # distinct reviews mentioning this aspect
    positive: int
    negative: int
    negative_ratio: float     # negative / (positive + negative); 0.0 when empty
    cells: dict[str, AspectCell] = field(default_factory=dict)


@dataclass
class CorpusSummary:
    """Final aggregator output for one batch of reviews."""
    n_reviews: int
    n_reviews_with_tuples: int
    n_tuples: int
    aspects: dict[str, AspectSummary]
    sentiment_overall: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "n_reviews": self.n_reviews,
            "n_reviews_with_tuples": self.n_reviews_with_tuples,
            "n_tuples": self.n_tuples,
            "sentiment_overall": dict(self.sentiment_overall),
            "aspects": {
                a: {
                    "aspect_category": s.aspect_category,
                    "total_mentions": s.total_mentions,
                    "n_reviews": s.n_reviews,
                    "positive": s.positive,
                    "negative": s.negative,
                    "negative_ratio": s.negative_ratio,
                    "cells": {
                        sent: {
                            "aspect_category": c.aspect_category,
                            "sentiment": c.sentiment,
                            "count": c.count,
                            "top_terms": c.top_terms,
                        }
                        for sent, c in s.cells.items()
                    },
                }
                for a, s in self.aspects.items()
            },
        }


# ---- Aggregator --------------------------------------------------------------

ReviewTuples = Sequence[PredictedTuple]


def aggregate(
    per_review_tuples: Iterable[ReviewTuples],
    *,
    top_k: int = 5,
    min_confidence: float = 0.0,
) -> CorpusSummary:
    """Aggregate a stream of per-review predictions into a CorpusSummary.

    Args:
        per_review_tuples: iterable where each element is the list of
            `PredictedTuple` for one review (possibly empty).
        top_k: keep at most this many distinct aspect_terms per cell.
        min_confidence: drop tuples below this confidence (already filtered
            inside the pipeline, but safe to re-apply).
    """
    cell_term_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    cell_counts: Counter[tuple[str, str]] = Counter()
    aspect_review_sets: dict[str, set[int]] = defaultdict(set)

    n_reviews = 0
    n_reviews_with_tuples = 0
    n_tuples = 0

    for review_idx, tuples in enumerate(per_review_tuples):
        n_reviews += 1
        kept_any = False
        for t in tuples:
            if t.confidence < min_confidence:
                continue
            if t.aspect_category not in ASPECTS or t.sentiment not in SENTIMENTS:
                continue
            key = (t.aspect_category, t.sentiment)
            cell_counts[key] += 1
            cell_term_counts[key][normalize_term(t.aspect_term)] += 1
            aspect_review_sets[t.aspect_category].add(review_idx)
            n_tuples += 1
            kept_any = True
        if kept_any:
            n_reviews_with_tuples += 1

    aspects: dict[str, AspectSummary] = {}
    sentiment_overall: Counter[str] = Counter()

    for asp in ASPECTS:
        pos = cell_counts.get((asp, "positive"), 0)
        neg = cell_counts.get((asp, "negative"), 0)
        total = pos + neg
        if total == 0:
            continue
        cells: dict[str, AspectCell] = {}
        for sent in SENTIMENTS:
            c = cell_counts.get((asp, sent), 0)
            if c == 0:
                continue
            terms = cell_term_counts[(asp, sent)].most_common(top_k)
            cells[sent] = AspectCell(
                aspect_category=asp,
                sentiment=sent,
                count=c,
                top_terms=terms,
            )
        aspects[asp] = AspectSummary(
            aspect_category=asp,
            total_mentions=total,
            n_reviews=len(aspect_review_sets[asp]),
            positive=pos,
            negative=neg,
            negative_ratio=neg / total if total else 0.0,
            cells=cells,
        )
        sentiment_overall["positive"] += pos
        sentiment_overall["negative"] += neg

    return CorpusSummary(
        n_reviews=n_reviews,
        n_reviews_with_tuples=n_reviews_with_tuples,
        n_tuples=n_tuples,
        aspects=aspects,
        sentiment_overall=dict(sentiment_overall),
    )


# ---- Convenience: rank aspects by "needs attention" --------------------------

def rank_by_priority(summary: CorpusSummary) -> list[AspectSummary]:
    """Order aspects from most-needs-attention to least.

    Heuristic: priority = negative_count * negative_ratio. Aspects with no
    mentions are skipped (they don't appear in `summary.aspects`).
    """
    return sorted(
        summary.aspects.values(),
        key=lambda s: (s.negative * s.negative_ratio, s.negative),
        reverse=True,
    )

"""Validate raw label dicts (e.g., Gemini output) against the project schema.

Pure functions — no I/O, no LLM calls. The weak-labeling pipeline calls
these to filter hallucinations and malformed rows before constructing
`Review` objects.

Drop reasons are surfaced via `ValidationStats` so the caller can log
per-reason counts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .label_schema import ASPECTS, SENTIMENTS
from .schema import Annotation, Review
from .span_align import find_cause_span


@dataclass
class ValidationStats:
    kept: int = 0
    dropped_bad_aspect: int = 0
    dropped_bad_sentiment: int = 0
    dropped_no_cause: int = 0
    dropped_empty_action: int = 0
    dropped_action_too_long: int = 0
    dropped_span_not_found: int = 0
    dropped_overlap: int = 0
    dropped_duplicate: int = 0
    dropped_schema: int = 0

    def merge(self, other: "ValidationStats") -> None:
        for k, v in other.__dict__.items():
            setattr(self, k, getattr(self, k) + v)

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def validate_annotation(
    raw: dict,
    review_text: str,
    *,
    max_action_words: int = 10,
    stats: ValidationStats | None = None,
) -> Annotation | None:
    """Validate one raw annotation dict. Returns Annotation or None."""
    s = stats or ValidationStats()

    aspect = raw.get("aspect")
    if aspect not in ASPECTS:
        s.dropped_bad_aspect += 1
        return None

    sentiment = raw.get("sentiment")
    if sentiment not in SENTIMENTS:
        s.dropped_bad_sentiment += 1
        return None

    cause = raw.get("cause") or raw.get("cause_text") or ""
    if not cause:
        s.dropped_no_cause += 1
        return None

    action = (raw.get("action") or "").strip()
    if not action:
        s.dropped_empty_action += 1
        return None
    if len(action.split()) > max_action_words:
        s.dropped_action_too_long += 1
        return None

    # Resolve span. Prefer explicit span if it actually matches; else search.
    span: tuple[int, int] | None = None
    raw_span = raw.get("cause_span")
    if isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
        try:
            a, b = int(raw_span[0]), int(raw_span[1])
            if 0 <= a < b <= len(review_text) and review_text[a:b] == cause:
                span = (a, b)
        except (TypeError, ValueError):
            span = None
    if span is None:
        span = find_cause_span(review_text, cause)
    if span is None:
        s.dropped_span_not_found += 1
        return None

    s_, e_ = span
    cause_text = review_text[s_:e_]
    try:
        return Annotation(
            aspect=aspect,
            sentiment=sentiment,
            cause_text=cause_text,
            cause_span=(s_, e_),
            action=action,
        )
    except Exception:
        s.dropped_schema += 1
        return None


def dedup_and_resolve_overlaps(
    anns: list[Annotation],
    stats: ValidationStats | None = None,
) -> list[Annotation]:
    """Remove exact duplicates and overlapping spans (keep first seen)."""
    s = stats or ValidationStats()
    seen_keys: set[tuple[str, str, int, int]] = set()
    out: list[Annotation] = []
    for ann in anns:
        key = (ann.aspect, ann.sentiment, ann.cause_span[0], ann.cause_span[1])
        if key in seen_keys:
            s.dropped_duplicate += 1
            continue
        if any(_spans_overlap(ann.cause_span, kept.cause_span) for kept in out):
            s.dropped_overlap += 1
            continue
        seen_keys.add(key)
        out.append(ann)
    return out


def validate_review(
    review_id: str,
    review_text: str,
    raw_annotations: list[dict],
    *,
    stats: ValidationStats | None = None,
) -> Review | None:
    s = stats or ValidationStats()
    candidates: list[Annotation] = []
    for raw in raw_annotations:
        if not isinstance(raw, dict):
            s.dropped_schema += 1
            continue
        ann = validate_annotation(raw, review_text, stats=s)
        if ann is not None:
            candidates.append(ann)
    deduped = dedup_and_resolve_overlaps(candidates, stats=s)
    s.kept += len(deduped)
    try:
        return Review(id=review_id, review=review_text, annotations=deduped)
    except Exception:
        s.dropped_schema += 1
        return None

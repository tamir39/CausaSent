"""Pydantic schema for the new ABSA annotation format.

New schema (post-pivot):
  review + annotations: [{aspect_term, aspect_term_span, aspect_category, sentiment}]

The old schema (cause_span, action) is dropped.
For data loading use dataset.load_absa_jsonl() which reads flat JSONL directly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .label_schema import ASPECTS, SENTIMENTS

Sentiment = Literal["positive", "negative"]


class Annotation(BaseModel):
    aspect_category: str
    aspect_term: str
    aspect_term_span: tuple[int, int] = Field(..., description="[start_char, end_char) into review")
    sentiment: Sentiment

    @field_validator("aspect_category")
    @classmethod
    def _aspect_in_taxonomy(cls, v: str) -> str:
        if v not in ASPECTS:
            raise ValueError(f"aspect_category {v!r} not in taxonomy {ASPECTS}")
        return v

    @field_validator("sentiment")
    @classmethod
    def _sentiment_binary(cls, v: str) -> str:
        if v not in SENTIMENTS:
            raise ValueError(f"sentiment {v!r} not in {SENTIMENTS}")
        return v

    @model_validator(mode="after")
    def _span_well_formed(self) -> "Annotation":
        s, e = self.aspect_term_span
        if not (0 <= s < e):
            raise ValueError(f"aspect_term_span malformed: {self.aspect_term_span}")
        return self


class Review(BaseModel):
    id: str
    review: str
    annotations: list[Annotation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _spans_match_text(self) -> "Review":
        for ann in self.annotations:
            s, e = ann.aspect_term_span
            if e > len(self.review):
                raise ValueError(f"aspect_term_span {ann.aspect_term_span} out of range for id={self.id}")
            substr = self.review[s:e]
            if substr != ann.aspect_term:
                raise ValueError(
                    f"span/term mismatch in id={self.id}: slice={substr!r} vs aspect_term={ann.aspect_term!r}"
                )
        return self


def load_reviews(path: str | Path) -> list[Review]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Review.model_validate(d) for d in data]


def save_reviews(reviews: list[Review], path: str | Path) -> None:
    payload = [r.model_dump() for r in reviews]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

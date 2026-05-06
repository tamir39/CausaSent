"""Pydantic schema for reviews and annotations (matches SPEC §1.2)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .label_schema import ASPECTS, SENTIMENTS

Sentiment = Literal["positive", "negative", "neutral"]


class Annotation(BaseModel):
    aspect: str
    sentiment: Sentiment
    cause_text: str
    cause_span: tuple[int, int] = Field(..., description="[start_char, end_char) into review")
    action: str

    @field_validator("aspect")
    @classmethod
    def _aspect_in_taxonomy(cls, v: str) -> str:
        if v not in ASPECTS:
            raise ValueError(f"aspect {v!r} not in taxonomy {ASPECTS}")
        return v

    @field_validator("sentiment")
    @classmethod
    def _sentiment_in_taxonomy(cls, v: str) -> str:
        if v not in SENTIMENTS:
            raise ValueError(f"sentiment {v!r} not in {SENTIMENTS}")
        return v

    @model_validator(mode="after")
    def _span_well_formed(self) -> "Annotation":
        s, e = self.cause_span
        if not (0 <= s < e):
            raise ValueError(f"cause_span malformed: {self.cause_span}")
        return self


class Review(BaseModel):
    id: str
    review: str
    annotations: list[Annotation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _spans_match_text(self) -> "Review":
        for ann in self.annotations:
            s, e = ann.cause_span
            if e > len(self.review):
                raise ValueError(f"cause_span {ann.cause_span} out of range for id={self.id}")
            substr = self.review[s:e]
            if substr != ann.cause_text:
                raise ValueError(
                    f"cause_span/cause_text mismatch in id={self.id}: "
                    f"slice={substr!r} vs cause_text={ann.cause_text!r}"
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

"""Print distributional stats for any reviews JSON.

Usage:
    python scripts/dataset_stats.py data/gold/train.json
    python scripts/dataset_stats.py data/processed/train.json --json out.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from src.data.schema import load_reviews


def _percentiles(values: list[int], qs: tuple[int, ...] = (10, 25, 50, 75, 95)) -> dict[int, int]:
    if not values:
        return {q: 0 for q in qs}
    s = sorted(values)
    out = {}
    for q in qs:
        idx = max(0, min(len(s) - 1, int(round((q / 100) * (len(s) - 1)))))
        out[q] = s[idx]
    return out


def compute_stats(path: str | Path) -> dict:
    reviews = load_reviews(path)
    n = len(reviews)
    aug = sum(1 for r in reviews if r.id.startswith("aug-"))
    review_lens = [len(r.review) for r in reviews]
    span_lens: list[int] = []
    aspects: Counter[str] = Counter()
    sentiments: Counter[str] = Counter()
    n_with_cause = 0

    for r in reviews:
        if r.annotations:
            n_with_cause += 1
        for a in r.annotations:
            aspects[a.aspect] += 1
            sentiments[a.sentiment] += 1
            span_lens.append(a.cause_span[1] - a.cause_span[0])

    return {
        "path": str(path),
        "count": n,
        "real": n - aug,
        "augmented": aug,
        "aug_ratio": (aug / n) if n else 0.0,
        "with_cause_pct": (n_with_cause / n) if n else 0.0,
        "review_len_chars": {
            "mean": round(statistics.fmean(review_lens), 1) if review_lens else 0,
            "percentiles": _percentiles(review_lens),
        },
        "cause_span_len_chars": {
            "mean": round(statistics.fmean(span_lens), 1) if span_lens else 0,
            "percentiles": _percentiles(span_lens),
        },
        "aspect_distribution": dict(aspects.most_common()),
        "sentiment_distribution": dict(sentiments.most_common()),
        "total_annotations": sum(aspects.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="Optional path to write stats as JSON.")
    args = ap.parse_args()

    stats = compute_stats(args.path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()

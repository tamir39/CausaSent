"""
Combine gold + Tiki aspect_term drafts into a single ABSA dataset, then make
a stratified 80/10/10 train/val/test split for the new schema.

Inputs:
    data/interim/gold_aspect_term_draft.jsonl
    data/interim/tiki_aspect_term_draft.jsonl

Outputs:
    data/processed/ate/train.json
    data/processed/ate/val.json
    data/processed/ate/test.json
    data/processed/ate/stats.json

Per-record schema (in train/val/test):
    {
      "id": "<id>",
      "source": "gold" | "tiki",
      "review": "<text>",
      "annotations": [
        {
          "aspect_term": "...",
          "aspect_term_span": [s, e],
          "aspect_category": "...",
          "sentiment": "positive" | "negative"
        }, ...
      ]
    }

The split is stratified by the dominant (aspect_category, sentiment) of each
review so each class lands across train/val/test proportionally.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

GOLD = Path("data/interim/gold_aspect_term_draft.jsonl")
TIKI = Path("data/interim/tiki_aspect_term_draft.jsonl")
OUT_DIR = Path("data/processed/ate")
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def group_by_review(records: list[dict]) -> list[dict]:
    """Collapse annotation-level records back into review-level records."""
    by_id: dict[str, dict] = {}
    for r in records:
        rid = r["id"]
        if rid not in by_id:
            by_id[rid] = {
                "id": rid,
                "source": r.get("source", "unknown"),
                "review": r["review"],
                "annotations": [],
            }
        by_id[rid]["annotations"].append({
            "aspect_term": r["aspect_term"],
            "aspect_term_span": r["aspect_term_span"],
            "aspect_category": r["aspect_category"],
            "sentiment": r["sentiment"],
        })
    return list(by_id.values())


def dominant_class(annotations: list[dict]) -> tuple[str, str]:
    """Pick the most common (aspect, sentiment) pair for stratification."""
    counts = Counter((a["aspect_category"], a["sentiment"]) for a in annotations)
    return counts.most_common(1)[0][0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    gold = load_jsonl(GOLD)
    tiki = load_jsonl(TIKI)
    print(f"Gold annotations: {len(gold)}")
    print(f"Tiki annotations: {len(tiki)}")

    reviews = group_by_review(gold) + group_by_review(tiki)
    print(f"Combined reviews: {len(reviews)} (after grouping)")

    # Stratified split — bucket by dominant class
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in reviews:
        buckets[dominant_class(r["annotations"])].append(r)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []

    for cell, rows in buckets.items():
        rng.shuffle(rows)
        n = len(rows)
        n_tr = int(n * TRAIN_FRAC)
        n_va = int(n * VAL_FRAC)
        train.extend(rows[:n_tr])
        val.extend(rows[n_tr:n_tr + n_va])
        test.extend(rows[n_tr + n_va:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    splits = {"train": train, "val": val, "test": test}
    for name, rows in splits.items():
        (OUT_DIR / f"{name}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Stats
    def aspect_counts(rows: list[dict]) -> dict:
        c: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})
        for r in rows:
            for a in r["annotations"]:
                c[a["aspect_category"]][a["sentiment"]] += 1
        return {k: dict(v) for k, v in c.items()}

    stats = {
        "total_reviews": len(reviews),
        "total_annotations": sum(len(r["annotations"]) for r in reviews),
        "by_source": {
            "gold": len([r for r in reviews if r["source"] == "gold"]),
            "tiki": len([r for r in reviews if r["source"] == "tiki"]),
        },
        "splits": {
            name: {
                "reviews": len(rows),
                "annotations": sum(len(r["annotations"]) for r in rows),
                "aspect_x_sentiment": aspect_counts(rows),
            }
            for name, rows in splits.items()
        },
    }
    (OUT_DIR / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSplits:")
    for name, rows in splits.items():
        anns = sum(len(r["annotations"]) for r in rows)
        print(f"  {name:6s} {len(rows):5d} reviews / {anns:5d} annotations")
    print(f"\nWrote: {OUT_DIR}/{{train,val,test}}.json + stats.json")


if __name__ == "__main__":
    main()

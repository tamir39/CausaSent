"""
Build final training dataset by merging CausaSent gold + Tiki annotations.
Produces 80/10/10 stratified splits by aspect_category x sentiment.

Inputs:
  data/interim/gold_aspect_term.jsonl        (from merge_corrections.py)
  data/interim/tiki_aspect_term_draft.jsonl  (from annotate_tiki.py)

Outputs:
  data/processed/train.jsonl
  data/processed/val.jsonl
  data/processed/test.jsonl
  data/processed/dataset_stats.json

Usage:
    python scripts/build_training_set.py
    python scripts/build_training_set.py --use-draft   # use draft (before human review)
    python scripts/build_training_set.py --no-tiki     # gold only
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")

GOLD_FINAL = INTERIM / "gold_aspect_term.jsonl"
GOLD_DRAFT = INTERIM / "gold_aspect_term_draft.jsonl"
TIKI_DRAFT = INTERIM / "tiki_aspect_term_draft.jsonl"

VALID_ASPECTS = {
    "delivery", "packaging", "product_quality",
    "price", "customer_service", "usability", "appearance",
}
VALID_SENTIMENTS = {"positive", "negative"}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    return records


def normalize(rec: dict, source_tag: str) -> dict | None:
    aspect = rec.get("aspect_category") or rec.get("aspect")
    sentiment = rec.get("sentiment")
    review = rec.get("review", "")
    term = rec.get("aspect_term", "")
    span = rec.get("aspect_term_span")

    if aspect not in VALID_ASPECTS:
        return None
    if sentiment not in VALID_SENTIMENTS:
        return None
    if not review or not term:
        return None
    if not span or len(span) != 2:
        return None

    s, e = span
    valid = isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(review)

    return {
        "id": rec.get("id", ""),
        "ann_idx": rec.get("ann_idx", 0),
        "review": review,
        "aspect_term": term,
        "aspect_term_span": [s, e],
        "aspect_category": aspect,
        "sentiment": sentiment,
        "source": source_tag,
        "valid_span": valid,
        "human_corrected": rec.get("human_corrected", False),
    }


def stratified_split(
    records: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        key = rec["aspect_category"] + "_" + rec["sentiment"]
        groups[key].append(rec)

    train, val, test = [], [], []
    for group in groups.values():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            train.extend(group)
        else:
            train.extend(group[:n_train])
            val.extend(group[n_train: n_train + n_val])
            test.extend(group[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_stats(name: str, records: list[dict]) -> None:
    from collections import Counter
    ac = Counter(r["aspect_category"] for r in records)
    sc = Counter(r["sentiment"] for r in records)
    src = Counter(r.get("source", "?") for r in records)
    print(f"\n  {name}: {len(records)} records")
    for asp, n in sorted(ac.items()):
        bar = "#" * (n // 20)
        print(f"    {asp:<20} {n:>5}  {bar}")
    print(f"    sentiment: {dict(sc)}")
    print(f"    source:    {dict(src)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-draft", action="store_true")
    ap.add_argument("--no-tiki", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    args = ap.parse_args()

    gold_path = GOLD_DRAFT if args.use_draft else GOLD_FINAL
    if not gold_path.exists():
        if not args.use_draft and GOLD_DRAFT.exists():
            print(f"NOTE: {GOLD_FINAL} not found — falling back to draft")
            gold_path = GOLD_DRAFT
        else:
            raise SystemExit(f"ERROR: Gold data not found at {gold_path}")

    print(f"Loading gold from {gold_path}")
    raw_gold = load_jsonl(gold_path)
    gold = [r for r in (normalize(rec, "gold") for rec in raw_gold) if r]
    print(f"  Gold: {len(raw_gold)} raw -> {len(gold)} valid")

    all_records = list(gold)

    if not args.no_tiki:
        if TIKI_DRAFT.exists() and TIKI_DRAFT.stat().st_size > 0:
            print(f"Loading Tiki from {TIKI_DRAFT}")
            raw_tiki = load_jsonl(TIKI_DRAFT)
            tiki = [r for r in (normalize(rec, "tiki") for rec in raw_tiki) if r]
            print(f"  Tiki: {len(raw_tiki)} raw -> {len(tiki)} valid")
            all_records.extend(tiki)
        else:
            print(f"WARNING: {TIKI_DRAFT} empty — building gold-only")
            print("  Run 'python scripts/annotate_tiki.py' to annotate Tiki data first")

    print(f"\nTotal: {len(all_records)} records")

    train, val, test = stratified_split(
        all_records, args.train_ratio, args.val_ratio, args.seed
    )

    write_jsonl(train, PROCESSED / "train.jsonl")
    write_jsonl(val, PROCESSED / "val.jsonl")
    write_jsonl(test, PROCESSED / "test.jsonl")

    print_stats("Train", train)
    print_stats("Val", val)
    print_stats("Test", test)

    stats = {
        "total": len(all_records),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "gold": len(gold),
        "tiki": len(all_records) - len(gold),
    }
    (PROCESSED / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nWrote splits to {PROCESSED}/")
    print(f"  train: {len(train)}  val: {len(val)}  test: {len(test)}")
    print("\nNext: upload Kaggle notebook for model training")


if __name__ == "__main__":
    main()

"""Batch inference: CSV/JSON of reviews → aggregated summary + action recs.

CLI:
    python -m src.inference.batch \
        --input reviews.csv \
        --review-col review \
        --ckpt checkpoints/phobert/best.pt \
        --output outputs/analysis.json \
        --top-k 5 \
        [--no-llm]      # skip Gemini, use template fallback
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from .action_llm import generate_actions, template_actions
from .aggregate import CorpusSummary, aggregate
from .pipeline import CausaSentPipeline, PredictedTuple


def _load_reviews(path: Path, review_col: str | None) -> list[tuple[str, str]]:
    """Return list of (id, review_text). `id` falls back to row index."""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [(str(r.get("id", i)), r["review"]) for i, r in enumerate(data)]
    # CSV / TSV
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    out: list[tuple[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        col = review_col or _guess_review_column(reader.fieldnames or [])
        if col is None:
            raise ValueError(
                f"Could not detect review column in {path.name}. "
                f"Headers: {reader.fieldnames}. Pass --review-col explicitly."
            )
        for i, row in enumerate(reader):
            text = (row.get(col) or "").strip()
            if not text:
                continue
            rid = str(row.get("id") or i)
            out.append((rid, text))
    return out


def _guess_review_column(headers: list[str]) -> str | None:
    candidates = ("review", "content", "comment", "text", "review_text", "nội_dung")
    lower = {h.lower(): h for h in headers}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def _predict_stream(
    pipeline: CausaSentPipeline,
    reviews: Iterable[tuple[str, str]],
) -> tuple[list[dict], list[list[PredictedTuple]]]:
    per_review_records: list[dict] = []
    per_review_tuples: list[list[PredictedTuple]] = []
    for rid, text in reviews:
        tuples = pipeline(text)
        per_review_tuples.append(tuples)
        per_review_records.append({
            "id": rid,
            "review": text,
            "tuples": [t.to_dict() for t in tuples],
        })
    return per_review_records, per_review_tuples


def run(
    *,
    input_path: Path,
    ckpt: Path,
    output_path: Path,
    review_col: str | None = None,
    top_k: int = 5,
    use_llm: bool = True,
    min_confidence: float = 0.0,
) -> dict:
    reviews = _load_reviews(input_path, review_col)
    pipeline = CausaSentPipeline(phobert_ckpt=str(ckpt), min_confidence=min_confidence)
    per_review, tuples_stream = _predict_stream(pipeline, reviews)
    summary: CorpusSummary = aggregate(tuples_stream, top_k=top_k, min_confidence=min_confidence)
    actions = (
        generate_actions(summary) if use_llm else template_actions(summary)
    )
    result = {
        "input": str(input_path),
        "n_reviews": len(reviews),
        "summary": summary.to_dict(),
        "actions": [a.to_dict() for a in actions],
        "per_review": per_review,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV/TSV/JSON of reviews")
    ap.add_argument("--review-col", default=None, help="Column name (CSV/TSV only)")
    ap.add_argument("--ckpt", default="checkpoints/phobert/best.pt")
    ap.add_argument("--output", default="outputs/analysis.json")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--no-llm", action="store_true", help="skip Gemini, use templates")
    args = ap.parse_args()

    res = run(
        input_path=Path(args.input),
        ckpt=Path(args.ckpt),
        output_path=Path(args.output),
        review_col=args.review_col,
        top_k=args.top_k,
        use_llm=not args.no_llm,
        min_confidence=args.min_confidence,
    )
    print(
        f"Analyzed {res['n_reviews']} reviews → "
        f"{res['summary']['n_tuples']} tuples → "
        f"{len(res['actions'])} actions"
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

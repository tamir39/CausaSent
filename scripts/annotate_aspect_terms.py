"""
Orchestrate Claude-based annotation of aspect_term + aspect_term_span for the
CausaSent gold dataset.

This file is now a DOC + thin CLI wrapper around the actual pipeline, which is
split across several scripts so each step can be inspected / resumed
independently:

    1. prepare_annotation_batches.py
         → Splits non-neutral gold annotations into
           data/interim/batches/batch_NNN.json
           (default 100 records/batch)

    2. (subagent annotation step — run via Claude Code)
         → For each batch, spawn a subagent that reads
           data/interim/batches/batch_NNN.json and writes
           data/interim/batches/output/batch_NNN_annotated.jsonl
         Subagent prompt template lives in scripts/SUBAGENT_PROMPT.md.

    3. verify_annotation_batch.py
         → Sanity-check that review[start:end] == aspect_term for every
           record in an annotated batch.

    4. merge_annotation_batches.py
         → Combine all annotated batches into
           data/interim/gold_aspect_term_draft.jsonl
         (drops any record whose span verification fails)

    5. build_validation_csv.py
         → Stratified-sample 300 records to
           data/interim/validation_sample.csv for human review.

    6. merge_corrections.py
         → Apply human corrections from validation_sample.csv back to the
           draft and emit gold_aspect_term_validated.jsonl
         (per-record source flag: llm_only | human_validated | human_corrected)

Why Claude (not Gemini) for annotation: see memory file
`feedback_llm_routing.md`. Gemini is reserved for the demo's action-generation
path only — every other LLM step in the project uses Claude.

This script no longer performs annotation directly; it just prints the
recommended invocation order.
"""
from __future__ import annotations


STEPS = [
    ("Split non-neutral gold annotations into batches",
     "python scripts/prepare_annotation_batches.py --batch-size 100"),
    ("Annotate each batch via subagent (orchestrated by Claude Code)",
     "# spawned interactively — see scripts/SUBAGENT_PROMPT.md"),
    ("Verify one batch's spans",
     "python scripts/verify_annotation_batch.py \\\n"
     "    --batch data/interim/batches/batch_001.json \\\n"
     "    --annotated data/interim/batches/output/batch_001_annotated.jsonl"),
    ("Merge all annotated batches into one draft",
     "python scripts/merge_annotation_batches.py"),
    ("Build validation CSV (300 records, stratified)",
     "python scripts/build_validation_csv.py --n 300"),
    ("[Human step] Edit data/interim/validation_sample.csv",
     "# manual review — fill your_correction / correct_start / correct_end"),
    ("Apply corrections and compute LLM-agreement rate",
     "python scripts/merge_corrections.py"),
]


def main() -> None:
    print("CausaSent — aspect_term annotation pipeline")
    print("=" * 60)
    for i, (desc, cmd) in enumerate(STEPS, start=1):
        print(f"\nStep {i}. {desc}")
        print(f"  $ {cmd}")
    print("\nOutputs land in data/interim/. Final artifact:")
    print("  data/interim/gold_aspect_term_validated.jsonl")


if __name__ == "__main__":
    main()

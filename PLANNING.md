# 🧭 CausaSent — PLANNING.md

> Operational planning document. Pairs with `PRD.md` (the *what*) and `SPEC.md` (the original brief).
> This file is the day-to-day map for engineers and AI agents working in this repo.

---

## 1. Architecture Summary

CausaSent is a **two-stage ML pipeline**: a token-classification model extracts structured tuples, a seq2seq model generates actions per tuple. There is no web service in MVP — only a CLI training/eval flow and a Gradio demo.

```
┌──────────────────────────────────────────────────────┐
│                Raw Vietnamese review                 │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │ VnCoreNLP word-seg   │
            └──────────┬───────────┘
                       │  segmented words
                       ▼
            ┌──────────────────────┐
            │ PhoBERT-large encoder│
            └──────┬─────────┬─────┘
                   │         │
        ┌──────────▼──┐  ┌───▼─────────┐
        │ Head A      │  │ Head B      │
        │ aspect+sent │  │ cause BIO   │
        │ BIO (43)    │  │ (3)         │
        └──────────┬──┘  └───┬─────────┘
                   │         │
                   └────┬────┘
                        ▼
            ┌──────────────────────┐
            │ BIO decoder          │
            │ → (aspect, sentiment,│
            │    cause_span)       │
            └──────────┬───────────┘
                       │  one prompt per tuple
                       ▼
            ┌──────────────────────┐
            │ mT5-base (action gen)│
            └──────────┬───────────┘
                       ▼
       (aspect, sentiment, cause_span, action) tuples
```

**Core principles (binding):**

1. **Closed-set taxonomy.** Aspects come from a fixed list of 7. The model is wrong by definition if it predicts anything else; downstream code never invents new aspects.
2. **Char-span correctness.** Every annotation/output `cause_span` MUST satisfy `review[start:end] == cause_text`. Validated in `Review` Pydantic model.
3. **Two-head separation.** Aspect-sentiment and cause are predicted by *separate* linear heads on a shared encoder — never combined into one label space.
4. **Word-boundary alignment.** Mismatched spans expand outward to the nearest VnCoreNLP word edges, never inward.
5. **No silent failure in weak labels.** `weak_label.py` drops a review if Gemini output fails JSON parse, taxonomy check, or substring check — partial garbage is worse than missing data.

---

## 2. Module Catalog

| Module                | Responsibility                                          | Key files                                       |
| --------------------- | ------------------------------------------------------- | ----------------------------------------------- |
| `src.data`            | Schema, label space, segmentation, span alignment, datasets, weak labeling, raw normalization. | `schema.py`, `label_schema.py`, `span_align.py`, `segmenter.py`, `dataset.py`, `weak_label.py`, `crawler.py` |
| `src.models`          | PhoBERT two-head tagger, mT5 wrapper.                   | `phobert_tagger.py`, `mt5_action.py`            |
| `src.train`           | Training entry points for both models.                  | `train_phobert.py`, `train_mt5.py`              |
| `src.eval`            | Token + entity F1, ROUGE-L, human-eval CSV dump.        | `eval_phobert.py`, `eval_mt5.py`                |
| `src.inference`       | BIO decode → tuples; end-to-end pipeline.               | `decode.py`, `pipeline.py`                      |
| `src.demo`            | Gradio app.                                             | `app.py`                                        |
| `scripts/`            | Dataset split, plumbing utilities.                      | `split_dataset.py`, `sample_data.json`          |
| `configs/`            | YAML hyperparameters per model.                         | `phobert.yaml`, `mt5.yaml`                      |

---

## 3. Folder Structure

```
CausaSent/
├── configs/                # Per-model YAML
├── data/
│   ├── raw/                # Raw JSONL (gitignored)
│   ├── processed/          # Weak-labeled / split JSON (gitignored)
│   └── gold/               # Hand-curated test (gitignored)
├── checkpoints/            # Model weights (gitignored)
├── outputs/                # Eval CSVs, generated artifacts (gitignored)
├── scripts/                # CLI utilities
├── src/
│   ├── data/
│   ├── models/
│   ├── train/
│   ├── eval/
│   ├── inference/
│   └── demo/
├── PRD.md                  # local-only
├── PLANNING.md             # tracked
├── TASKS.md                # local-only
├── CLAUDE.md               # local-only
├── SPEC.md                 # tracked (original brief)
├── README.md               # tracked
├── requirements.txt
└── .gitignore
```

---

## 4. Data Model

### 4.1 Review (canonical JSON)

```json
{
  "id": "string",
  "review": "string",
  "annotations": [
    {
      "aspect": "delivery | packaging | product_quality | price | customer_service | usability | appearance",
      "sentiment": "positive | negative | neutral",
      "cause_text": "string",
      "cause_span": [start_char_inclusive, end_char_exclusive],
      "action": "string (Vietnamese, imperative, ≤10 words)"
    }
  ]
}
```

Validated by `src.data.schema.Review` (Pydantic v2). `cause_span` slice MUST equal `cause_text`.

### 4.2 BIO label spaces

| Head | Label space | Size |
| ---- | ----------- | ---- |
| A — aspect+sentiment | `O` + `B-{ASPECT_CODE}-{SENT_CODE}` × 7 × 3 + `I-...` × 7 × 3 | 43 |
| B — cause | `O`, `B-CAUSE`, `I-CAUSE` | 3 |

Aspect codes: DEL, PACK, QUAL, PRICE, SVC, USE, APP. Sentiment codes: POS, NEG, NEU.

### 4.3 mT5 input/output template

```
Input  : "aspect: {aspect}\nsentiment: {sentiment}\ncause: {cause}\nreview: {review}"
Output : "{action}"
```

Defined in `src.data.dataset.ACTION_INPUT_TEMPLATE`.

---

## 5. Training Plan

### 5.1 PhoBERT tagger

- Shared encoder: `vinai/phobert-large`.
- Two parallel `nn.Linear(hidden, n_labels)` heads.
- Loss = `CE(asp) + cause_loss_weight * CE(cause)`, both with `IGNORE_INDEX=-100` masking sub-words/specials.
- Class weights: inverse-frequency over the train set, computed once at startup.
- Hyperparameters: `lr=2e-5`, `bs=16`, `epochs=5`, `max_len=128`, AdamW, linear warmup 10%.

### 5.2 mT5 action generator

- Base model: `google/mt5-base`.
- Trained on `(aspect, sentiment, cause, review) → action` pairs derived from gold + pseudo data.
- Hyperparameters: `lr=3e-5`, `bs=8`, `epochs=5`, `max_input_len=128`, `max_output_len=32`, AdamW.
- Generation: 4-beam search, length penalty 1.0.

### 5.3 Sequencing

1. Curate dataset (weak label → human refine → split).
2. Train PhoBERT to convergence; pick best by val loss.
3. (Optional) Use PhoBERT predictions on train to enlarge the action-generation training set with `(predicted_aspect, predicted_sentiment, predicted_cause, review) → gold_action` pairs — improves robustness to imperfect upstream extraction.
4. Train mT5.
5. Evaluate both on the gold test set.

---

## 6. Evaluation

| Component         | Metric                                | Where                          |
| ----------------- | ------------------------------------- | ------------------------------ |
| Tagger (token)    | seqeval token-level F1, per head      | `src/eval/eval_phobert.py`     |
| Tagger (entity)   | (start_word, end_word, base_tag) F1   | `src/eval/eval_phobert.py`     |
| Action generation | ROUGE-L (mean over test set)          | `src/eval/eval_mt5.py`         |
| Action generation | Human eval (50–100 samples): "reasonable" + "actionable" 0/1 | CSV dump → manual scoring      |

---

## 7. Decision Log

- **PhoBERT-large + mT5-base** committed per SPEC §9 — no benchmarking other architectures.
- **Two parallel heads**, not a flat combined label space (would explode to ~50 classes and tangle aspect-sentiment learning with cause learning).
- **No span overlap** assumed — each token belongs to at most one aspect.
- **Gemini 2.5 Flash** for weak labels (cheap, reliable JSON mode).
- **Hybrid data sourcing** — public Vietnamese review datasets + small targeted manual export. No automated scraping.
- **VnCoreNLP word-seg** kept (PhoBERT was pretrained on word-segmented text — skipping costs ~2-3 F1).
- **Span misalignment policy:** expand outward to nearest word boundary; never silently truncate.
- **Action style guide:** imperative, verb-first, ≤10 Vietnamese words.
- **Class imbalance:** inverse-frequency weighted CE on the aspect-sentiment head.
- **Demo:** Gradio (free, fast to ship, embeds in HF Spaces if needed).

---

## 8. Roadmap

| Phase | Focus                                     | Deliverable                                                             |
| ----- | ----------------------------------------- | ----------------------------------------------------------------------- |
| **1** | MVP pipeline + dataset + demo             | Trained checkpoints, gold dataset, Gradio demo, eval report.            |
| **2** | Active learning + domain expansion        | Re-fine-tuned models, broader aspect taxonomy, larger gold corpus.      |
| **3** | Production-ready inference                | Distilled tagger, ONNX export, optional service wrapper.                |

---

## Invariants (DO NOT VIOLATE)

- PRD.md is the source of truth for requirements.
- Aspect taxonomy is closed (7 values). Never extend without a PRD update.
- `cause_span` MUST satisfy `review[start:end] == cause_text` at every layer (data, training, inference, eval).
- Tagger has exactly two heads — do not merge into one combined label space.
- VnCoreNLP word-segmentation is run upstream of PhoBERT for both training and inference (consistency requirement).
- No automated scraping of Shopee / TikTok Shop.
- Gemini API keys via env var only — never committed.

---

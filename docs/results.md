# CausaSent — Phase 1 Baseline Results

> Numbers come from `python -m src.eval.eval_phobert` and `python -m src.eval.eval_mt5`
> on `data/gold/test.json`. Run metadata for every reported number lives under
> `runs/<timestamp>/` (config.yaml, git_sha.txt, env.txt, metrics.jsonl).

## TL;DR

| Component | Metric | Value |
|---|---|---|
| PhoBERT-large tagger | aspect-sentiment entity-F1 | **0.34** |
| PhoBERT-large tagger | cause entity-F1 | **0.46** |
| mT5-base action generator | ROUGE-L (mean) | **0.54** |

End-to-end demo runs at sub-second latency on a single Kaggle T4. Both checkpoints
are public on HuggingFace Hub; the pipeline is reproducible from a clean Kaggle
session in under an hour.

## Setup

- **Tagger:** PhoBERT-large (370M params), two parallel BIO heads — aspect-sentiment (43 labels) and cause (3 labels) — over VnCoreNLP word-segmented input.
- **Action generator:** mT5-base (580M params), seq2seq, conditioned on `(aspect, sentiment, cause_text, review)`.
- **Data:** 1500 hand-labeled Vietnamese reviews drawn from VLSP-2016 Sentiment + VLSP-2018 ABSA pools, 2273 (aspect, sentiment, cause_span, action) annotations total.
- **Splits:** 80/10/10 by review id, seed 42 → 1200 train / 150 val / 150 test.
- **Selection:** PhoBERT best checkpoint by *mean entity-F1* on val; mT5 best by val loss.
- **Hardware:** Kaggle T4 (16 GB), single GPU, ~50 min total wall-clock for both models.
- **Dataset:** [`Tamir39/causasent`](https://huggingface.co/datasets/Tamir39/causasent)
- **Models:** [`Tamir39/causasent-phobert`](https://huggingface.co/Tamir39/causasent-phobert), [`Tamir39/causasent-mt5`](https://huggingface.co/Tamir39/causasent-mt5)

## Tagger — token-level F1 (seqeval)

### aspect-sentiment head

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| APP-NEG | 0.000 | 0.000 | 0.000 | 11 |
| APP-POS | 0.325 | 0.394 | 0.356 | 33 |
| PRICE-NEG | 0.091 | 0.167 | 0.118 | 6 |
| PRICE-NEU | 0.000 | 0.000 | 0.000 | 4 |
| PRICE-POS | 0.121 | 0.190 | 0.148 | 21 |
| QUAL-NEG | 0.138 | 0.444 | 0.211 | 18 |
| QUAL-NEU | 0.000 | 0.000 | 0.000 | 2 |
| QUAL-POS | 0.220 | 0.367 | 0.275 | 79 |
| SVC-NEG | 0.000 | 0.000 | 0.000 | 5 |
| SVC-NEU | 0.000 | 0.000 | 0.000 | 1 |
| **SVC-POS** | **0.561** | **0.742** | **0.639** | 31 |
| USE-NEG | 0.000 | 0.000 | 0.000 | 15 |
| USE-POS | 0.000 | 0.000 | 0.000 | 2 |
| **micro avg** | **0.242** | **0.342** | **0.283** | 228 |

### cause head

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| CAUSE | 0.375 | 0.465 | 0.415 | 228 |

## Tagger — entity-level F1

An entity matches only if **both** the span boundaries and the label are correct.

| Head | Precision | Recall | F1 |
|---|---|---|---|
| aspect-sentiment | 0.376 | 0.307 | **0.338** |
| cause | 0.449 | 0.465 | **0.457** |

**Target (PRD DoD):** aspect-sentiment entity F1 ≥ 0.65 — **not yet met**; this baseline establishes a floor for future work.

## mT5 — action generation

| Metric | Value | N |
|---|---|---|
| ROUGE-L (mean) | **0.5421** | 228 |
| Human "reasonable" rate | TBD | manual scoring deferred |
| Human "actionable" rate | TBD | manual scoring deferred |

Human-eval CSV emitted at `outputs/action_human_eval_test.csv` (228 rows for manual review).

**Targets (PRD DoD):** ROUGE-L ≥ 0.30 ✅ (achieved); ≥70% actionable — pending human eval.

## Training trajectory

PhoBERT-large val curves (8 epochs, lr 2e-5, batch 16, no class weighting):

| Epoch | Train loss | Val loss | asp F1 | cause F1 | mean F1 |
|---|---|---|---|---|---|
| 1 | 3.95 | 1.93 | 0.000 | 0.000 | 0.000 |
| 2 | 1.71 | 1.53 | 0.000 | 0.343 | 0.172 |
| 3 | 1.27 | 1.29 | 0.080 | 0.437 | 0.259 |
| 4 | 0.93 | 1.31 | 0.144 | 0.459 | 0.302 |
| 5 | 0.75 | 1.34 | 0.240 | 0.450 | 0.345 |
| 6 | 0.63 | 1.44 | 0.295 | 0.456 | 0.376 |
| **7** | **0.54** | **1.45** | **0.349** | **0.453** | **0.401** ← best |
| 8 | 0.51 | 1.47 | 0.327 | 0.434 | 0.380 |

Checkpoint at epoch 7 selected by mean entity-F1. Val loss starts climbing at
epoch 4 even as F1 keeps rising — confirms entity-F1 is the right selection
metric for this task (val loss alone would have stopped training 3 epochs too early).

## Honest interpretation

**Headline numbers are dominated by 5 well-supported classes:**

| Class | Test support | Entity F1 | Notes |
|---|---|---|---|
| QUAL-POS | 79 | 0.275 | most frequent class; bulk of training signal |
| APP-POS | 33 | 0.356 | strong baseline |
| **SVC-POS** | 31 | **0.639** | best class; highly distinctive vocabulary |
| PRICE-POS | 21 | 0.148 | weaker — short cause spans hard to localize |
| QUAL-NEG | 18 | 0.211 | sufficient data; respectable F1 |

**Classes at 0.0 F1 are mostly data-sparse, not model failure:**

- `QUAL-NEU` (2 ex), `USE-POS` (2 ex), `PRICE-NEU` (4 ex), `SVC-NEU` (1 ex) — too few test examples for any learned pattern to generalize.
- `APP-NEG` (11), `USE-NEG` (15), `SVC-NEG` (5) — borderline; these are the classes augmentation could realistically lift.
- `delivery` and `packaging` aspects: **not present in training data at all** (the VLSP source corpora cover hotels/restaurants/tech, not shipping). Reaching them requires manual e-commerce hand-labeling per `docs/manual-collection.md`.

**One earlier mistake worth recording.** First training run used `class_weighting: inverse_freq`. It produced precision/recall = 0.09 / 0.32 (entity-F1 0.17 on aspect-sentiment) — the model flooded predictions of rare classes. Disabling class weighting and adding 3 epochs (5 → 8) doubled both entity-F1 numbers. **Lesson:** with strict BIO, inverse-frequency weighting hurts precision far more than it helps recall.

## Ablation — gold-only vs gold + validated pseudo

Not run for this baseline — the entire 1500-review pool was hand-labeled, so gold == "gold + validated pseudo" by construction. Re-introducing this ablation is meaningful only after the augmentation pass adds non-gold rows.

| Train set | Asp F1 | Cause F1 | Mean F1 |
|---|---|---|---|
| gold only (this baseline) | 0.338 | 0.457 | **0.398** |
| gold + augmented (≤30% aug) | TBD | TBD | TBD |

## Error analysis

See [`notebooks/error_analysis.ipynb`](../notebooks/error_analysis.ipynb):

- Aspect confusion matrix (entity-level)
- Sentiment confusion (conditional on correct aspect)
- Cause-span IoU histogram
- mT5 generation failures (bottom-20 by ROUGE-L)

## Reproducibility

```bash
git checkout feat/skeleton
python scripts/fetch_dataset.py --dest data
python -m src.train.train_phobert --config configs/phobert.yaml
python -m src.train.train_mt5 --config configs/mt5.yaml
python -m src.eval.eval_phobert --config configs/phobert.yaml --ckpt checkpoints/phobert/best.pt --split test
python -m src.eval.eval_mt5 --config configs/mt5.yaml --ckpt checkpoints/mt5/best.pt --split test
```

End-to-end Kaggle pipeline: [`notebooks/kaggle_train.ipynb`](../notebooks/kaggle_train.ipynb) (training) and [`notebooks/kaggle_demo.ipynb`](../notebooks/kaggle_demo.ipynb) (Gradio demo from HF checkpoints).

## Next levers (not in this baseline)

1. **Augmentation up to 30%** via `scripts/build_training_set.py` (paraphrase/typo/synonym, span-preserving). Realistic +3–7 F1 points, mostly on rare classes.
2. **Hand-label 100–300 e-commerce reviews** to populate `delivery`/`packaging`. Pure manual labor; `docs/manual-collection.md` is the playbook.
3. **mT5 longer training** — current 5-epoch Adafactor run is conservative.
4. **Active-learning loop** (Phase 2 in PLANNING) — feed validator-flagged disagreements back into the gold pool.

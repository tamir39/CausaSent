# CausaSent

**Vietnamese aspect-based sentiment analysis with aggregated action recommendations for e-commerce reviews.**

> Final-year Data Mining project. The system turns a batch of free-text Vietnamese reviews into a corpus-level summary and a prioritised list of business actions for the seller — not just per-review sentiment labels.

[**🤗 Dataset**](https://huggingface.co/datasets/Tamir39/causasent-ate-v2) · [**🤗 Model**](https://huggingface.co/Tamir39/causasent-phobert-ate) · [**Paper (IEEE)**](paper/main.tex) · [**Báo cáo cuối kỳ (VN)**](report/)

---

## What it does

For each batch of Vietnamese product reviews:

1. **Extract** `(aspect_term, aspect_category, sentiment)` tuples per review using a fine-tuned PhoBERT-large with two parallel classification heads.
2. **Aggregate** N reviews into a frequency-ranked grid over `(aspect, sentiment)` cells.
3. **Recommend** prioritised Vietnamese imperative actions for the seller via a single Gemini-2.5-Flash call grounded on the aggregated summary.

The closed 7-class aspect taxonomy is `delivery`, `packaging`, `product_quality`, `price`, `customer_service`, `usability`, `appearance`. Sentiment is binary (`positive` / `negative`).

## Headline results (held-out test split, 719 reviews / 1,023 annotations)

| Metric                          | Value  |
|---------------------------------|-------:|
| ATE entity-level F1             | **0.328** (P 0.21, R 0.80) |
| Sentiment macro F1              | **0.876** |
| Sentiment binary F1 (negative)  | **0.901** |
| Inter-annotator agreement       | **91.0%** (300-record stratified sample) |

Per-aspect F1: delivery 0.43 · packaging 0.37 · price 0.32 · appearance 0.31 · product_quality 0.30 · customer_service 0.24 · usability 0.23.

The model is deliberately tuned recall-first on ATE; the aggregation step absorbs span-level false positives via frequency ranking. See [`paper/main.tex`](paper/main.tex) §5.4 for the design argument.

## Quick start (Windows)

```powershell
# clone + install backend deps
uv pip install -r requirements.txt
uv pip install -r apps/api/requirements.txt
uv pip install python-dotenv python-docx py-vncorenlp

# fetch the trained checkpoint (1.48 GB) from HF Hub
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download(repo_id='Tamir39/causasent-phobert-ate', \
                    local_dir='checkpoints/phobert')"

# run the full stack — FastAPI + Next.js + SSE streaming
.\run.bat
```

`run.bat` opens the FastAPI backend in a new window (port 8000) and the Next.js frontend in the current terminal (port 3000). The backend lazily loads PhoBERT-large (~10–15 s the first call), then `/analyze-stream` emits Server-Sent Events as each review is processed.

Optional environment:

```
GEMINI_API_KEY=...            # enables Gemini action generation
                              # (without it, deterministic Vietnamese templates are used)
CAUSASENT_PYTHON=...          # explicit Python interpreter for run.bat
JAVA_HOME=...                 # required by VnCoreNLP word-seg
```

### Demo workflow

1. Open <http://localhost:3000>.
2. Wait for the **API live** badge in the form panel.
3. Paste reviews (one per line) or upload a CSV — a 30-row sample is downloadable from the form.
4. Click **Phân tích** → dashboard opens immediately and streams in:
   - KPI strip with live counters (reviews processed, aspect tuples, % negative).
   - Aspect-coverage radar that fills as data arrives.
   - Aspect breakdown grid with top complaints / praises chips.
   - Action queue that lights up once Gemini returns.
   - Reviews table with client-side search, sentiment filters, and click-to-expand.

## Architecture

```
                  ┌─────────────────────────┐
                  │ Review (Vietnamese text)│
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │  VnCoreNLP word-seg     │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │  PhoBERT-large encoder  │
                  └──────┬──────────┬───────┘
                         │          │
            ┌────────────┘          └────────────┐
            ▼                                    ▼
    ┌──────────────────┐               ┌──────────────────┐
    │  Head A — ATE    │               │  Head B — sent.  │
    │  15-way BIO      │               │  binary @ B-tok  │
    └────────┬─────────┘               └─────────┬────────┘
             │                                   │
             └────────────────┬──────────────────┘
                              ▼
                  ┌─────────────────────────┐
                  │  BIO decoder            │
                  │  → tuples per review    │
                  └────────────┬────────────┘
                               │  N reviews
                               ▼
                  ┌─────────────────────────┐
                  │  Aggregator             │
                  │  (aspect, sentiment)    │
                  │  cells + top-K terms    │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │  Gemini 2.5 Flash       │
                  │  (single structured     │
                  │  call, JSON output)     │
                  └────────────┬────────────┘
                               ▼
                Prioritised Vietnamese actions
```

### Hard invariants (never violated by any layer)

- `review[aspect_term_span[0]:aspect_term_span[1]] == aspect_term` byte-for-byte (NFC-normalised).
- `aspect_category` is one of 7 closed values; never extended at inference.
- Tagger has exactly two parallel heads; ATE BIO and sentiment are never merged into one combined label space.
- VnCoreNLP word-segmentation runs upstream of PhoBERT in **both** training and inference.
- LLM is called once per batch for action generation — never per review.

## Dataset

`Tamir39/causasent-ate-v2` on the Hugging Face Hub. 7,066 reviews / 10,307 annotations stratified 80/10/10 by dominant `(aspect_category, sentiment)`.

| Source        | Reviews | Annotations | Notes |
|---------------|--------:|------------:|-------|
| Gold (manual) | 1,132   | 2,205       | Subset of VLSP 2016/2018 + smartphone gold, neutral dropped, span re-extracted via Claude pipeline |
| Tiki silver   | 5,934   | 8,102       | Tiki product reviews from a midterm collection, span added via the same pipeline |
| **Total**     | **7,066** | **10,307** | 99.84% pass span byte-match verification |

A 300-record stratified validation pass by an independent Claude judge agreed with the originally-extracted span in **273 / 300 cases (91.00%)**. Per-aspect breakdown in `data/interim/agreement_report.json`.

## Model

PhoBERT-large fine-tune with two parallel classification heads + supervised contrastive auxiliary:

- **Head A** — 15-way BIO over `O` + `B-/I-` for each of 7 aspect categories, applied at every word-level position.
- **Head B** — binary `positive` / `negative`, applied only at B-token positions; all other positions masked with `IGNORE_INDEX`.
- **Auxiliary loss** — supervised contrastive on B-token encoder representations, grouped by `aspect_category` (λ = 0.1).

Training: 8 epochs on a single Kaggle P100 (≈ 50 min), batch size 16, AdamW lr 2e-5, linear warmup 10%. Class-weighted CE on both heads (inverse frequency). Best checkpoint by `mean(F1_ATE_entity, Acc_sentiment)` on val (epoch 6, mean = 0.603).

The training notebook ([`notebooks/kaggle_train_absa.ipynb`](notebooks/kaggle_train_absa.ipynb)) clones the repo, pulls the HF dataset, runs `python -m src.train.train_phobert --config configs/phobert.yaml`, and pushes the checkpoint back to `Tamir39/causasent-phobert-ate`.

## Repository layout

```
configs/         training + model YAML
data/            raw / processed / gold / interim (mostly gitignored)
  HF_README_ATE_V2.md         # dataset card pushed to HF
src/
  data/          schema, label space, segmenter, dataset, span align
  models/        PhoBERT two-head tagger
  train/         training loop
  eval/          seqeval + sklearn metrics
  inference/     pipeline · aggregator · action LLM · batch CLI
  demo/          Gradio demo (legacy, kept working)
apps/
  api/           FastAPI inference service (lazy-loaded, SSE streaming)
  web/           Next.js 14 PWA — App Router, Tailwind, Recharts
paper/           IEEE conference-format paper (.tex + .bib)
report/          Vietnamese final-year project report (.docx + generator)
notebooks/       Kaggle training notebook
scripts/         CLI utilities + run.bat helpers
samples/         reviews_demo.csv (30-row sample)
tests/           pytest — aggregator + action LLM template fallback
run.bat          single-command launcher (API + web)
```

## Stack

- **Modelling** — PyTorch, Hugging Face Transformers, PhoBERT-large, VnCoreNLP, seqeval, scikit-learn.
- **Backend** — FastAPI, uvicorn, Server-Sent Events, python-dotenv.
- **Frontend** — Next.js 14 (App Router), TypeScript, Tailwind CSS, Recharts. PWA manifest + service-worker offline shell.
- **LLM** — Claude (annotation + evaluation, off the demo path); Gemini 2.5 Flash (action generation only, on the demo path).
- **Tooling** — uv for Python deps, Kaggle for training, Hugging Face Hub for distribution.

## Tests

```bash
python -m pytest tests/test_aggregate.py tests/test_action_llm.py -v
```

13 unit tests covering the aggregator, term normalisation, priority bucketing, template fallback, and Gemini response parsing. No model load required — fast.

## Paper & report

- [`paper/main.tex`](paper/main.tex) + [`paper/references.bib`](paper/references.bib) — IEEE conference-format paper (English). Compile with `pdflatex && bibtex && pdflatex × 2`. All numeric results filled, narrative complete.
- [`report/CausaSent_BaoCao.docx`](report/CausaSent_BaoCao.docx) — Vietnamese-language extended report for the school. Generated by [`scripts/build_report.py`](scripts/build_report.py) — edit content there and re-run rather than editing the `.docx` directly.

## License

CC-BY-SA-4.0 for the dataset and trained checkpoint. Code under the same license unless a file states otherwise. Built on `vinai/phobert-large` (MIT) and CC-BY-SA-4.0 portions of VLSP / UIT-ViSD4SA.

If you use this work, please cite the dataset card and the project.

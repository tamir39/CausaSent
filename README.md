# CausaSent

Causal & actionable sentiment analysis for Vietnamese reviews.

For each review, the system produces tuples of:

```
(aspect, sentiment, cause_span, action)
```

See [SPEC.md](SPEC.md) for the full specification.

## Architecture

- **Task 1+2 — joint aspect + sentiment + cause extraction.** PhoBERT-large with two parallel token-classification heads:
  - Head A: aspect-sentiment BIO (`B-DEL-NEG`, `I-PACK-POS`, …, `O`)
  - Head B: cause BIO (`B-CAUSE`, `I-CAUSE`, `O`)
- **Task 3 — action generation.** mT5-base seq2seq, conditioned on `(aspect, sentiment, cause, review)`.

## Layout

```
configs/         training & model hyperparameters
data/            raw / processed / gold splits (gitignored)
src/
  data/          schema, weak labeling, span alignment, datasets
  models/        PhoBERT two-head tagger, mT5 action generator
  train/         training entry points
  eval/          token/entity F1, ROUGE-L
  inference/     end-to-end pipeline
  demo/          Gradio app
scripts/         one-off CLI utilities
```

## Quick start

```bash
pip install -r requirements.txt
python -m src.train.train_phobert --config configs/phobert.yaml
python -m src.train.train_mt5     --config configs/mt5.yaml
python -m src.demo.app                            # launch Gradio demo
```

## Decisions locked

| # | Decision |
|---|----------|
| BIO heads | Two parallel heads (aspect-sentiment / cause) |
| Overlap | Disallowed — one aspect per token |
| Weak labels | Gemini 2.5 Flash, JSON mode |
| Data sources | Public VN datasets + small targeted crawl |
| Sentiment | `{positive, negative, neutral}` |
| Tokenization | VnCoreNLP word-seg → PhoBERT |
| Span misalign | Expand to nearest word boundary |
| Action style | Imperative, verb-first, ≤10 words |
| Loss | Weighted cross-entropy (class imbalance) |
| Demo | Gradio |

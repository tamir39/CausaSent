---
license: cc-by-sa-4.0
task_categories:
  - token-classification
  - text2text-generation
language:
  - vi
size_categories:
  - 1K<n<10K
tags:
  - vietnamese
  - reviews
  - aspect-based-sentiment
  - cause-extraction
  - e-commerce
pretty_name: CausaSent
---

# CausaSent — Vietnamese Review Causal Sentiment Dataset

Per-review tuples of `(aspect, sentiment, cause_span, action)` for Vietnamese e-commerce reviews.

## Schema

Each row in `gold/{train,val,test}.json` is:

```json
{
  "id": "string",
  "review": "Ship lâu nhưng đóng gói đẹp",
  "annotations": [
    {
      "aspect": "delivery",
      "sentiment": "negative",
      "cause_text": "Ship lâu",
      "cause_span": [0, 8],
      "action": "Rút ngắn thời gian giao hàng"
    },
    {
      "aspect": "packaging",
      "sentiment": "positive",
      "cause_text": "đóng gói đẹp",
      "cause_span": [15, 27],
      "action": "Duy trì cách đóng gói"
    }
  ]
}
```

Hard invariants on every annotation (validated at load time):

- `review[cause_span[0]:cause_span[1]] == cause_text` (exact substring; Python codepoint indexing).
- `aspect ∈ {delivery, packaging, product_quality, price, customer_service, usability, appearance}`.
- `sentiment ∈ {positive, negative, neutral}`.
- Within a review, no two `cause_span`s overlap.
- `action` is imperative, verb-first, ≤10 Vietnamese words.

## Splits

| Split | Reviews | Annotations |
|-------|---------|-------------|
| train | TBD     | TBD         |
| val   | TBD     | TBD         |
| test  | TBD     | TBD         |

Numbers populated after `scripts/split_dataset.py` runs.

## Annotation process

1. Raw reviews collected from public Vietnamese review datasets (UIT-VSFC, ViSFD, Foody) plus a small manual export of Shopee / Google Maps reviews (ToS-safe, copy-paste only).
2. Weak-labeled with Gemini 2.5 Flash via `src/data/weak_label.py` (closed-taxonomy prompt, JSON mode, exact-substring constraint).
3. Validated by `src/data/validate.py` (drops hallucinated spans, bad aspects, duplicates, overlaps).
4. Manually refined to ≥1000 gold samples per [`docs/annotation-guide.md`](https://github.com/tamir39/causa-sent/blob/develop/docs/annotation-guide.md) via the Gradio review UI in `src/tools/review_ui.py`.
5. 80/10/10 splits by review id with seed 42.

## Intended use

- Train aspect-sentiment + cause extraction models (e.g., PhoBERT-large two-head tagger).
- Train action generation models (e.g., mT5-base) conditioned on extracted tuples.
- Benchmark fine-grained ABSA + causal-rationale tasks on Vietnamese.

## License

Released under **CC-BY-SA-4.0**. Derivative works must share-alike. Underlying public review corpora retain their original licenses; respect those when redistributing.

## Citation

```bibtex
@misc{causasent2026,
  title  = {CausaSent: Causal and Actionable Sentiment Analysis for Vietnamese Reviews},
  author = {Phí Vương Tường Tâm},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/datasets/Tamir39/causasent}}
}
```

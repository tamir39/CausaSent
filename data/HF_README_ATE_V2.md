---
license: cc-by-sa-4.0
task_categories:
  - token-classification
language:
  - vi
size_categories:
  - 1K<n<10K
tags:
  - vietnamese
  - reviews
  - aspect-term-extraction
  - aspect-based-sentiment
  - e-commerce
  - absa
pretty_name: CausaSent ATE v2
---

# CausaSent ATE v2 — Vietnamese E-commerce Aspect Term Extraction + ABSA

Vietnamese e-commerce reviews labeled with **aspect terms (span)**, **aspect category**, and **binary sentiment** — combined from a small manually-annotated gold pool (hotel + restaurant + smartphone reviews) and a larger LLM-labeled silver pool (Tiki product reviews).

## Schema

Each row in `train.json` / `val.json` / `test.json` is:

```json
{
  "id": "tiki-14075703",
  "source": "tiki",
  "review": "Hàng đóng gói không cẩn thận, hộp méo mó, rách nhiều.",
  "annotations": [
    {
      "aspect_term": "Hàng",
      "aspect_term_span": [0, 4],
      "aspect_category": "appearance",
      "sentiment": "negative"
    },
    {
      "aspect_term": "đóng gói",
      "aspect_term_span": [5, 13],
      "aspect_category": "packaging",
      "sentiment": "negative"
    }
  ]
}
```

**Hard invariants:**

- `review[aspect_term_span[0]:aspect_term_span[1]] == aspect_term` byte-for-byte.
- `aspect_category` ∈ {`delivery`, `packaging`, `product_quality`, `price`, `customer_service`, `usability`, `appearance`} — closed 7-class taxonomy.
- `sentiment` ∈ {`positive`, `negative`} — neutral was dropped (only 2.7% of gold, too few to train).
- `source` ∈ {`gold`, `tiki`}.

## Splits

| Split | Reviews | Annotations |
|---|---|---|
| train | 5,647 | 8,251 |
| val | 700 | 1,033 |
| test | 719 | 1,023 |
| **total** | **7,066** | **10,307** |

Stratified 80/10/10 by dominant `(aspect_category, sentiment)` per review.

Aspect coverage in train (annotation count):

| aspect_category | count |
|---|---|
| appearance | 2,789 |
| product_quality | 2,133 |
| delivery | 982 |
| packaging | 896 |
| price | 922 |
| customer_service | 424 |
| usability | 105 |

Sentiment leans negative for `appearance` / `packaging` (Tiki complaints) and positive for `price` / `customer_service`.

## Data sources

- **Gold (1,132 reviews, 2,205 annotations):** subset of CausaSent original manual annotations (drop `neutral`, drop `cause_span`/`action` fields). Schema converted via Claude-assisted aspect-term span extraction; 3 NFD-encoding edge cases dropped. See [Tamir39/causasent](https://huggingface.co/datasets/Tamir39/causasent) for the pre-pivot 4-tuple version.
- **Tiki silver (5,934 reviews, 8,102 annotations):** Vietnamese Tiki product reviews originally labeled with `aspect_category` + `sentiment` by an LLM (mid-term project); we added `aspect_term_span` via the same Claude pipeline. 14 records dropped on span verification.

## Annotation pipeline

Both sources passed through the same Claude-based annotation pipeline (`scripts/annotate_aspect_terms.py` orchestrator → `scripts/SUBAGENT_PROMPT.md` per-batch worker → `scripts/verify_annotation_batch.py` byte-match check → `scripts/merge_annotation_batches.py`). 99.86% of gold and 99.83% of Tiki survived span verification.

A stratified sample of 300 gold records was independently re-validated; inter-annotator agreement ≈ 91% (judged term matches LLM's pick).

## Intended use

Joint aspect-term extraction + aspect-category-aware sentiment classification on Vietnamese e-commerce text. Suitable for:
- Training PhoBERT/XLM-R BIO span taggers with auxiliary sentiment head.
- Comparing fine-tuned models vs. zero-shot LLM baselines on Vietnamese ATE.
- ABSA-informed action recommendation pipelines (per the paper this dataset accompanies).

## License & ToS

Released under CC-BY-SA-4.0. The Tiki subset comes from publicly-displayed product reviews; we redistribute only the text and our derived annotations. If you use this dataset, please cite the original CausaSent project.

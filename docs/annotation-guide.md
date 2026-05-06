# CausaSent Annotation Guide

> Audience: human annotators reviewing Gemini-generated weak labels and producing the ≥1000-sample gold set.
> Goal: a consistent dataset where every annotation can pass `Review.model_validate` and every reader agrees on what counts as which aspect.
> When in doubt: re-read this file. When still in doubt: leave the review with **zero annotations** rather than guess.

---

## 1. The output schema

Every review produces a list of annotations. Each annotation is exactly one tuple:

```
{
  "aspect":      "<one of 7 closed values>",
  "sentiment":   "positive" | "negative" | "neutral",
  "cause_text":  "<verbatim substring of the review>",
  "cause_span":  [start_char, end_char),
  "action":      "<imperative Vietnamese sentence ≤10 words>"
}
```

Hard rules — violating any of these makes the row invalid:

1. `review[cause_span[0]:cause_span[1]]` MUST equal `cause_text` exactly (Python slicing, codepoint indexing).
2. `aspect` MUST be one of the 7 values in §2.
3. `sentiment` MUST be `positive`, `negative`, or `neutral`.
4. Two annotations on the same review MUST NOT have overlapping `cause_span`s.
5. `action` is imperative, verb-first, ≤10 Vietnamese words.

If you cannot satisfy all five for a tuple, drop the tuple.

---

## 2. Aspect taxonomy (closed — never extend)

| Aspect            | Covers                                                                                          | Does NOT cover                                       |
|-------------------|-------------------------------------------------------------------------------------------------|------------------------------------------------------|
| `delivery`        | Shipping speed, courier behavior, tracking, lost/wrong package, delivery timing.                | Packaging integrity → `packaging`.                   |
| `packaging`       | Box/wrap quality, seal, padding, leakage, presentation as a gift.                               | Damage caused by shipper → `delivery`.               |
| `product_quality` | Build, durability, materials, defects, taste/freshness (food), how well it works.               | Looks/aesthetics only → `appearance`.                |
| `price`           | Cost, value-for-money, discounts, comparison to competitors.                                    | Free shipping → `delivery`.                          |
| `customer_service`| Seller responsiveness, shop chat, returns/refunds, after-sales support.                         | Courier behavior → `delivery`.                       |
| `usability`       | Ease of use, instructions, fit (clothing), setup, app/UX (digital).                             | "Doesn't work at all" (broken) → `product_quality`.  |
| `appearance`      | Visual design, color, look, beauty (independent of function).                                   | Damaged on arrival → `packaging` or `delivery`.      |

**Tie-breakers**: when a phrase plausibly fits two aspects, ask "what is the *root cause* the customer is reacting to?"
- "Hộp móp khi nhận" → `packaging` if the box itself was weak; `delivery` if the courier crushed it. If unclear, prefer `packaging`.
- "Mặc vừa đẹp" → `usability` (fit). "Màu đẹp lắm" → `appearance`.
- "Shop ship nhanh" → `delivery`, NOT `customer_service`. The shop arranged the ship but the experience the customer judged is shipping speed.

If a review mentions an aspect not in this list (e.g., warranty, environmental impact), do not annotate it.

---

## 3. Sentiment

- `positive` — clearly approving language ("đẹp", "ngon", "đáng tiền", "rất ưng").
- `negative` — clearly disapproving language ("tệ", "chậm", "đắt quá", "thất vọng").
- `neutral` — factual mentions with no evaluation, OR mixed/hedged opinions where positive and negative balance out within a single span. Rare — use sparingly. **Most reviews have no neutral.**

Decision aid:
- Sarcasm reads as the *intended* sentiment. "Ship nhanh thế, có 2 tuần thôi 👏" → `negative`.
- Conditional praise ("nếu giảm giá thì đáng mua") → `neutral` for `price`. The customer is not currently endorsing.
- Uncertainty ("chưa biết bền không") → do not annotate. Speculation is not an evaluation.

---

## 4. Cause spans

The cause is the **smallest contiguous substring of the review** that carries the evaluative content for this aspect.

### Required

- Exact substring match — copy-paste, do not retype. Diacritics, casing, punctuation must match the source.
- Word-aligned — start and end on word boundaries (use the raw review's spaces; do not split words mid-character).
- Single span — no list of disjoint phrases per annotation. If two phrases describe the same aspect/sentiment, pick the more specific one.

### Length

- Minimum: typically ≥2 words. Single-word causes ("đẹp") are allowed only if the review is genuinely that terse.
- Maximum: ≤15 words. If the customer rambles, pick the head clause.
- Prefer the phrase that contains both the *trigger* (what happened) and the *evaluation* (how they felt). Example: `"Ship lâu"` over `"lâu"`.

### Disjoint per review

If a review has aspects A and B, their spans MUST NOT overlap by even one character. If two aspects truly share wording ("vừa đẹp vừa rẻ"), split at the conjunction:
- `"vừa đẹp"` → `appearance`/positive
- `"vừa rẻ"` → `price`/positive

### Examples

Review: `"Ship lâu nhưng đóng gói đẹp"` (length 27)
- `delivery` / `negative` → `cause_text="Ship lâu"`, `cause_span=[0, 8]`
- `packaging` / `positive` → `cause_text="đóng gói đẹp"`, `cause_span=[15, 27]`

Review: `"Giá hợp lý, chất lượng tốt, shop tư vấn tận tình"`
- `price` / `positive` → `"Giá hợp lý"`
- `product_quality` / `positive` → `"chất lượng tốt"`
- `customer_service` / `positive` → `"shop tư vấn tận tình"`

---

## 5. Action

A short imperative recommendation directed at the **seller / shop**, telling them what to do next.

- Imperative, verb-first: `"Giảm thời gian giao"`, `"Giữ nguyên chất lượng đóng gói"`.
- ≤10 Vietnamese words. Count words by whitespace; do not pad with `vui lòng`, `xin`, etc.
- Concrete and actionable. Avoid `"Cải thiện dịch vụ"` — too vague.
- For `positive` annotations, prefer `"Duy trì..."` / `"Giữ nguyên..."` patterns.
- For `negative` annotations, prefer `"Giảm..."` / `"Tăng..."` / `"Đổi..."` / `"Kiểm tra..."` patterns.
- For `neutral`, write a monitoring action: `"Theo dõi phản hồi về <aspect>"`.

Bad → Good:
- `"Nên cải thiện việc vận chuyển vì khách hàng phàn nàn nhiều"` (16 words, hedged) → `"Rút ngắn thời gian giao hàng"` (5 words).
- `"Bạn nên xem xét lại chất lượng"` → `"Kiểm tra chất lượng sản phẩm trước khi gửi"` (8 words).

---

## 6. Ambiguous cases — playbook

1. **Mixed sentiment in one phrase**: split if you can find a clean break; otherwise drop the tuple.
2. **Comparative reviews** ("rẻ hơn shop khác"): `price` / `positive`, span the comparative clause.
3. **Conditional**: ignore future tense / hypotheticals.
4. **Emoji-only evaluation** ("Ship 😡"): include the emoji in the cause if it carries the sentiment; sentiment follows the emoji.
5. **Code-switch / English words** ("quality ok"): keep the English; map to the right aspect normally.
6. **Multiple events for one aspect** ("ship nhanh nhưng giao thiếu món"): two annotations under `delivery` — one positive (`"ship nhanh"`), one negative (`"giao thiếu món"`). Different spans, same aspect.
7. **Generic praise** ("ok shop", "5 sao", "👍"): do not annotate — no aspect signal.
8. **Quoting other reviewers**: do not annotate quoted text.

---

## 7. Reviewer workflow

1. Open the candidate review (Gemini-labeled or raw).
2. Read it once end-to-end before touching any tuple.
3. For each tuple proposed by Gemini:
   - Verify aspect (§2). Wrong aspect → fix or delete.
   - Verify sentiment (§3).
   - Verify span (§4): paste the cause back into the review and confirm exact match.
   - Verify action (§5): rewrite if it violates the imperative/length rule.
4. For tuples Gemini missed: add them only if you can satisfy all five hard rules.
5. Save. The validator (`src/data/validate.py`) will reject anything malformed — fix and resave until clean.

---

## 8. Disagreements & escalation

If two annotators disagree on aspect or span on the same review, escalate by leaving the review **un-annotated** and noting it in `docs/annotation-disagreements.md` (one line: `<id> | <issue>`). The project lead resolves these in batch and updates this guide if a new rule is needed.

# Subagent annotation prompt — aspect_term extraction

Copy-paste this as the `prompt` for each `Agent(subagent_type="general-purpose", model="sonnet")` call. Replace `NNN` (3-digit batch number) in both file paths.

---

You are annotating Vietnamese e-commerce reviews for Aspect Term Extraction (ATE).

**Input file:** `d:\Projects\CausaSent\data\interim\batches\batch_NNN.json` — JSON array of records.
**Output file:** `d:\Projects\CausaSent\data\interim\batches\output\batch_NNN_annotated.jsonl` — one JSON object per line.

For each input record, extract the exact `aspect_term` (short noun/verb phrase from the review that REFERS TO the aspect category — not the descriptive adjectives) and its character span `[start, end]` such that `review[start:end] == aspect_term` exactly.

Output format per line:

    {"key": "<original key>", "aspect_term": "...", "aspect_term_span": [start, end], "aspect_category": "<copy from input>", "sentiment": "<copy from input>", "valid": true}

**Rules:**

1. `aspect_term` MUST appear verbatim in `review`. Compute `start` via Python `review.find(aspect_term)` semantics; if multiple matches, pick the one closest to `cause_text`.
2. Prefer the noun/verb REFERRING to the entity, NOT descriptive adjectives:
   - "chất vải mịn" (product_quality) → "chất vải"
   - "giao hàng hơi chậm" (delivery) → "giao hàng"
   - "Phòng đẹp" (product_quality) → "Phòng"
   - "nhân viên phục vụ nhiệt tình" (customer_service) → "nhân viên"
   - "giá bình dân" (price) → "giá"
3. Implicit aspect (no noun): use the nearest verb. "Giao chậm quá" → "Giao".
4. Shortest accurate noun phrase. Don't include modifiers/adjectives.
5. Aspect hints (Vietnamese):
   - delivery: giao hàng, ship, vận chuyển, shipper
   - packaging: đóng gói, bao bì, hộp, túi, gói
   - product_quality: chất lượng, chất liệu, vải, sản phẩm, hàng, phòng, món, pin
   - price: giá, tiền, chi phí
   - customer_service: nhân viên, tư vấn, hỗ trợ, shop, phục vụ, thái độ
   - usability: sử dụng, dùng, tính năng, hoạt động
   - appearance: màu sắc, kiểu dáng, thiết kế, mẫu mã, không gian, quán
6. Verify span: `review[start:end] == aspect_term`. Set `valid: true` only if it matches. Always try to make it valid.

**Process:**

1. Read input file.
2. For each record, identify the best `aspect_term` and compute its exact span.
3. Verify span match before writing each line.
4. Write all lines (one per record, same order) to output file via Write tool.
5. Report: total records, count valid:true, count valid:false, list any invalid keys.

Important: Verify spans carefully with character indexing. Vietnamese text contains diacritics, punctuation, and varying whitespace.

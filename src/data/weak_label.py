"""Weak labeling via Gemini 2.5 Flash with JSON-mode output.

Reads raw reviews from a JSONL file (one `{"id":..., "review":...}` per line)
and writes a labeled JSON file matching `src.data.schema.Review`.

Usage:
    GEMINI_API_KEY=... python -m src.data.weak_label \\
        --in data/raw/reviews.jsonl \\
        --out data/processed/weak.json \\
        --limit 100
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .label_schema import ASPECTS, SENTIMENTS
from .schema import Annotation, Review, save_reviews
from .span_align import find_cause_span

PROMPT = """\
Bạn là chuyên gia phân tích review thương mại điện tử tiếng Việt.

Hãy trích xuất các bộ (aspect, sentiment, cause, action) từ review sau.

Quy tắc bắt buộc:
- aspect PHẢI thuộc danh sách: {aspects}
- sentiment ∈ {sentiments}
- cause PHẢI là một đoạn con (substring) chính xác của review, không được paraphrase.
- action ngắn gọn, dạng mệnh lệnh (verb-first), tối đa 10 từ tiếng Việt.
- Mỗi annotation chỉ thuộc đúng MỘT aspect. Không được trùng span giữa các aspect.
- Nếu review không nhắc tới aspect nào trong danh sách, trả về mảng rỗng.

Trả về JSON array thuần (không markdown, không text khác):
[
  {{"aspect": "...", "sentiment": "...", "cause": "...", "action": "..."}}
]

Review:
\"\"\"{review}\"\"\"
"""


def _build_prompt(review: str) -> str:
    return PROMPT.format(
        aspects=", ".join(ASPECTS),
        sentiments=", ".join(SENTIMENTS),
        review=review,
    )


def _call_gemini(client, prompt: str, model_name: str, max_retries: int = 3) -> str:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json", "temperature": 0.0},
            )
            return resp.text
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini call failed after {max_retries} retries: {last_err}")


def label_review(client, review_id: str, review_text: str, model_name: str) -> Review | None:
    raw = _call_gemini(client, _build_prompt(review_text), model_name)
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return None
    anns: list[Annotation] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        aspect = it.get("aspect")
        sentiment = it.get("sentiment")
        cause = it.get("cause", "")
        action = it.get("action", "").strip()
        if aspect not in ASPECTS or sentiment not in SENTIMENTS or not cause or not action:
            continue
        span = find_cause_span(review_text, cause)
        if span is None:
            continue
        s, e = span
        # Use the actual substring from the review (canonical casing).
        cause_text = review_text[s:e]
        try:
            anns.append(Annotation(
                aspect=aspect,
                sentiment=sentiment,
                cause_text=cause_text,
                cause_span=(s, e),
                action=action,
            ))
        except Exception:
            continue
    return Review(id=review_id, review=review_text, annotations=anns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set")

    import google.generativeai as genai  # lazy import

    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(args.model)

    reviews: list[Review] = []
    with open(args.in_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            row = json.loads(line)
            r = label_review(client, row["id"], row["review"], args.model)
            if r is not None:
                reviews.append(r)
            if (i + 1) % 25 == 0:
                print(f"[weak_label] processed {i + 1}, kept {len(reviews)}")

    save_reviews(reviews, args.out_path)
    print(f"[weak_label] wrote {len(reviews)} reviews → {args.out_path}")


if __name__ == "__main__":
    main()

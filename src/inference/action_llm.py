"""Generate Vietnamese business actions from an aggregated ABSA summary.

Two modes:
  - `generate_actions(summary)`        — calls Gemini once with a structured
                                         prompt and returns model-written actions.
  - `template_actions(summary)`        — deterministic offline fallback used
                                         when there's no API key, the call fails,
                                         or for unit tests.

Output schema (both paths):
  [{
    "aspect_category": "...",
    "sentiment": "positive|negative",
    "priority": "high|medium|low",
    "action": "Vietnamese imperative sentence (<= ~15 words)",
    "evidence_terms": ["term1", "term2", ...]
  }, ...]

Per `CLAUDE.md`: Gemini is ONLY used here (demo action generation). Everywhere
else in the codebase, Claude does the LLM work.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .aggregate import AspectSummary, CorpusSummary, rank_by_priority


ASPECT_LABEL_VI = {
    "delivery":         "Giao hàng",
    "packaging":        "Đóng gói",
    "product_quality":  "Chất lượng sản phẩm",
    "price":            "Giá cả",
    "customer_service": "Hỗ trợ khách hàng",
    "usability":        "Trải nghiệm sử dụng",
    "appearance":       "Hình thức / Ngoại quan",
}


@dataclass
class ActionRecommendation:
    aspect_category: str
    sentiment: str
    priority: str
    action: str
    evidence_terms: list[str]

    def to_dict(self) -> dict:
        return {
            "aspect_category": self.aspect_category,
            "sentiment": self.sentiment,
            "priority": self.priority,
            "action": self.action,
            "evidence_terms": self.evidence_terms,
        }


# ---- Priority bucketing ------------------------------------------------------

def _priority_for(asp: AspectSummary, sentiment: str, n_reviews: int = 1) -> str:
    """Bucket each (aspect, sentiment) cell into high/medium/low priority.

    Uses a continuous score that scales with batch size:
      - For negative cells:  score = negative * negative_ratio
        (penalises rare aspects with high neg-ratio that are likely noise,
        rewards genuinely-dominant complaints)
      - For positive cells:  score = positive * (1 - negative_ratio)
        (genuine strength worth doubling down on)

    Thresholds scale with `n_reviews` so the same batch ratios produce
    consistent priorities regardless of whether you analysed 30 or 3,000
    reviews — `max(absolute_floor, fraction_of_batch)` floor.
    """
    n = max(1, n_reviews)
    if sentiment == "negative":
        score = asp.negative * asp.negative_ratio
        if score >= max(5.0, 0.05 * n):
            return "high"
        if score >= max(2.0, 0.02 * n):
            return "medium"
        return "low"
    # positive
    score = asp.positive * max(0.0, 1.0 - asp.negative_ratio)
    if score >= max(10.0, 0.10 * n):
        return "high"
    if score >= max(5.0, 0.05 * n):
        return "medium"
    return "low"


# ---- Offline template fallback ----------------------------------------------

_NEG_TEMPLATES = {
    "delivery":         "Cải thiện tốc độ giao hàng cho {term}.",
    "packaging":        "Khắc phục lỗi đóng gói liên quan đến {term}.",
    "product_quality":  "Nâng cao chất lượng sản phẩm, ưu tiên {term}.",
    "price":            "Xem xét lại chiến lược giá cho {term}.",
    "customer_service": "Tập huấn đội hỗ trợ khách hàng về {term}.",
    "usability":        "Đơn giản hoá trải nghiệm liên quan đến {term}.",
    "appearance":       "Cải tiến ngoại quan sản phẩm, đặc biệt là {term}.",
}
_POS_TEMPLATES = {
    "delivery":         "Duy trì chất lượng giao hàng — điểm mạnh ở {term}.",
    "packaging":        "Giữ phong cách đóng gói hiện tại — khách khen {term}.",
    "product_quality":  "Tiếp tục đầu tư vào chất lượng {term}.",
    "price":            "Giữ mức giá hiện tại — khách đánh giá cao {term}.",
    "customer_service": "Khen thưởng đội hỗ trợ — khách nhắc tới {term}.",
    "usability":        "Quảng bá trải nghiệm dễ dùng quanh {term}.",
    "appearance":       "Đẩy mạnh hình ảnh sản phẩm — điểm cộng ở {term}.",
}


def _top_terms_str(asp: AspectSummary, sentiment: str, k: int = 3) -> tuple[str, list[str]]:
    cell = asp.cells.get(sentiment)
    if cell is None or not cell.top_terms:
        return ASPECT_LABEL_VI.get(asp.aspect_category, asp.aspect_category).lower(), []
    terms = [t for t, _ in cell.top_terms[:k]]
    return ", ".join(terms), terms


def template_actions(summary: CorpusSummary) -> list[ActionRecommendation]:
    """Deterministic, no-API fallback that always works."""
    out: list[ActionRecommendation] = []
    n_reviews = summary.n_reviews
    for asp in rank_by_priority(summary):
        for sentiment, cell in asp.cells.items():
            term_str, evidence = _top_terms_str(asp, sentiment)
            tmpl_table = _NEG_TEMPLATES if sentiment == "negative" else _POS_TEMPLATES
            tmpl = tmpl_table[asp.aspect_category]
            out.append(
                ActionRecommendation(
                    aspect_category=asp.aspect_category,
                    sentiment=sentiment,
                    priority=_priority_for(asp, sentiment, n_reviews),
                    action=tmpl.format(term=term_str),
                    evidence_terms=evidence,
                )
            )
    return out


# ---- Gemini-powered structured generation -----------------------------------

_GEMINI_PROMPT = """Bạn là chuyên gia phân tích phản hồi khách hàng cho sàn TMĐT Việt Nam.

Dưới đây là kết quả ABSA tổng hợp từ N reviews. Mỗi nhóm gồm `aspect_category`
(7 nhóm cố định), `sentiment` (positive/negative), số lần được nhắc và những
cụm từ (aspect_term) khách nhắc nhiều nhất.

Hãy đề xuất các HÀNH ĐỘNG KINH DOANH cụ thể, ngắn gọn, mệnh lệnh tiếng Việt
(≤ 15 từ mỗi câu). Mỗi (aspect, sentiment) có ý nghĩa khác nhau:
- negative → khắc phục / cải thiện điểm yếu
- positive → duy trì / khuếch trương điểm mạnh

QUY TẮC OUTPUT (rất quan trọng):
- Trả về thuần JSON, KHÔNG bọc trong markdown / code fence.
- Mỗi mục giữ nguyên `aspect_category` và `sentiment` từ input.
- `priority` ∈ {{"high","medium","low"}}.
- `action` viết bằng tiếng Việt, mệnh lệnh, không xưng "tôi"/"chúng tôi".
- `evidence_terms` là tối đa 3 cụm từ lấy từ `top_terms` của input.

INPUT:
{payload}

SCHEMA OUTPUT:
[
  {{
    "aspect_category": "...",
    "sentiment": "positive|negative",
    "priority": "high|medium|low",
    "action": "...",
    "evidence_terms": ["..."]
  }}
]
""".strip()


def _build_payload(summary: CorpusSummary) -> str:
    """Compact, model-friendly representation of the summary."""
    items = []
    for asp in rank_by_priority(summary):
        for sentiment, cell in asp.cells.items():
            items.append({
                "aspect_category": asp.aspect_category,
                "aspect_vi": ASPECT_LABEL_VI.get(asp.aspect_category, asp.aspect_category),
                "sentiment": sentiment,
                "count": cell.count,
                "negative_ratio_in_aspect": round(asp.negative_ratio, 3),
                "top_terms": [t for t, _ in cell.top_terms],
            })
    return json.dumps({
        "n_reviews": summary.n_reviews,
        "n_tuples": summary.n_tuples,
        "items": items,
    }, ensure_ascii=False, indent=2)


def _parse_response(raw: str) -> list[dict[str, Any]]:
    """Robust JSON extraction — Gemini occasionally wraps in ```json fences."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        # strip optional `json` language hint
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
        # remove trailing fence if any
        if s.endswith("```"):
            s = s[:-3].strip()
    return json.loads(s)


def generate_actions(
    summary: CorpusSummary,
    *,
    model_name: str = "gemini-2.5-flash",
    api_key: str | None = None,
    timeout_s: float = 30.0,
) -> list[ActionRecommendation]:
    """Call Gemini for structured actions; fall back to templates on any error.

    The Gemini key is read from `GEMINI_API_KEY` env var (or pass `api_key=`).
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return template_actions(summary)

    if not summary.aspects:
        return []

    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        return template_actions(summary)

    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(
            model_name,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.4,
            },
        )
        payload = _build_payload(summary)
        prompt = _GEMINI_PROMPT.format(payload=payload)
        resp = model.generate_content(prompt, request_options={"timeout": timeout_s})
        text = (resp.text or "").strip()
        if not text:
            return template_actions(summary)
        items = _parse_response(text)
    except Exception:  # noqa: BLE001 — defensive fallback by design
        return template_actions(summary)

    out: list[ActionRecommendation] = []
    valid_priorities = {"high", "medium", "low"}
    for it in items:
        if not isinstance(it, dict):
            continue
        asp = it.get("aspect_category")
        sent = it.get("sentiment")
        action = (it.get("action") or "").strip()
        if not asp or not sent or not action:
            continue
        priority = it.get("priority") or "medium"
        if priority not in valid_priorities:
            priority = "medium"
        ev = it.get("evidence_terms") or []
        if not isinstance(ev, list):
            ev = []
        out.append(
            ActionRecommendation(
                aspect_category=asp,
                sentiment=sent,
                priority=priority,
                action=action,
                evidence_terms=[str(x) for x in ev][:3],
            )
        )
    return out or template_actions(summary)

"""Gradio demo for the new ATE + ABSA pipeline.

Two-tab UI:
  Tab 1 — Phân tích 1 review: paste a review → highlighted aspect terms + tuples.
  Tab 2 — Phân tích nhiều review: paste/upload N reviews → aggregated dashboard
          + Gemini-generated action recommendations (with template fallback).
"""
from __future__ import annotations

import argparse
import html as _html
import os

import gradio as gr

from ..inference.action_llm import (
    ASPECT_LABEL_VI,
    generate_actions,
    template_actions,
)
from ..inference.aggregate import aggregate, rank_by_priority
from ..inference.pipeline import CausaSentPipeline


SENT_COLOR = {
    "positive": "#22c55e",
    "negative": "#ef4444",
}

PRIORITY_COLOR = {
    "high":   "#ef4444",
    "medium": "#f59e0b",
    "low":    "#64748b",
}


EXAMPLES_SINGLE = [
    "Ship lâu nhưng đóng gói đẹp, sản phẩm dùng tạm được, giá cũng ổn.",
    "Áo chất vải mịn, mặc thoáng, shop tư vấn nhiệt tình, giao nhanh.",
    "Hàng giao chậm 5 ngày, hộp móp méo, nhân viên trả lời cộc lốc, thất vọng.",
    "Giá hơi cao so với chất lượng, nhưng màu sắc và kiểu dáng thì ưng.",
]

EXAMPLES_BATCH = """Ship lâu nhưng đóng gói đẹp, sản phẩm dùng tạm được, giá cũng ổn.
Áo chất vải mịn, mặc thoáng, shop tư vấn nhiệt tình, giao nhanh.
Hàng giao chậm 5 ngày, hộp móp méo, nhân viên trả lời cộc lốc, thất vọng.
Giá hơi cao so với chất lượng, nhưng màu sắc và kiểu dáng thì ưng.
Sản phẩm chất lượng tốt, giao hàng nhanh, giá hợp lý.
Đóng gói cẩn thận, sản phẩm đẹp như hình, sẽ ủng hộ shop tiếp."""


def _highlight_review(review: str, tuples) -> str:
    """Wrap aspect_term spans with sentiment-colored underline."""
    if not tuples:
        return (
            '<div style="padding:0.6em 0.9em; line-height:1.6; font-size:1.05em;'
            ' background:#1e293b; border-radius:6px; color:#e2e8f0;">'
            f"{_html.escape(review)}</div>"
        )
    spans = sorted(
        ((t.aspect_term_span[0], t.aspect_term_span[1], t.sentiment) for t in tuples),
        key=lambda x: x[0],
    )
    out, cur = [], 0
    for s, e, sent in spans:
        if s < cur:
            continue
        out.append(_html.escape(review[cur:s]))
        color = SENT_COLOR.get(sent, "#fff")
        out.append(
            f'<span style="border-bottom:3px solid {color}; padding:0 2px;'
            f' background:{color}22;">{_html.escape(review[s:e])}</span>'
        )
        cur = e
    out.append(_html.escape(review[cur:]))
    return (
        '<div style="padding:0.8em 1em; line-height:1.8; font-size:1.05em;'
        ' background:#1e293b; border-radius:6px; color:#e2e8f0;">'
        + "".join(out)
        + "</div>"
    )


def _render_tuple_cards(tuples) -> str:
    if not tuples:
        return (
            '<div style="padding:1em; color:#94a3b8; font-style:italic;">'
            "Chưa trích xuất được aspect nào. Thử review dài hơn hoặc rõ ràng hơn."
            "</div>"
        )
    cards = []
    for i, t in enumerate(tuples, 1):
        color = SENT_COLOR.get(t.sentiment, "#a3a3a3")
        aspect_vi = ASPECT_LABEL_VI.get(t.aspect_category, t.aspect_category)
        cards.append(f'''
        <div style="border:1px solid #334155; border-left:4px solid {color};
                    border-radius:6px; padding:0.7em 1em; margin-bottom:0.6em;
                    background:#0f172a; color:#e2e8f0;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3em;">
            <div>
              <span style="font-weight:600;">#{i} · {_html.escape(aspect_vi)}</span>
              <span style="color:#94a3b8; font-size:0.85em; margin-left:0.4em;">
                ({_html.escape(t.aspect_category)})
              </span>
            </div>
            <span style="background:{color}; color:#0f172a; padding:0.1em 0.6em;
                         border-radius:999px; font-weight:600; font-size:0.8em;">
              {_html.escape(t.sentiment)}
            </span>
          </div>
          <div>
            <span style="color:#94a3b8; font-size:0.85em;">Aspect term:</span>
            <span style="border-bottom:2px solid {color}; padding:0 2px;">
              {_html.escape(t.aspect_term)}
            </span>
            <span style="color:#64748b; font-size:0.78em; margin-left:0.4em;">
              [{t.aspect_term_span[0]}, {t.aspect_term_span[1]})
            </span>
          </div>
          <div style="margin-top:0.3em; color:#64748b; font-size:0.78em;">
            confidence: {t.confidence:.2f}
          </div>
        </div>
        ''')
    return "".join(cards)


def _render_summary(summary) -> str:
    if not summary.aspects:
        return (
            '<div style="padding:1em; color:#94a3b8; font-style:italic;">'
            "Không trích xuất được aspect nào từ batch này.</div>"
        )
    rows = []
    for asp in rank_by_priority(summary):
        bar_pos = asp.positive
        bar_neg = asp.negative
        total = bar_pos + bar_neg
        pos_pct = 100 * bar_pos / total if total else 0
        neg_pct = 100 - pos_pct
        terms_lines = []
        for sent, cell in asp.cells.items():
            color = SENT_COLOR[sent]
            chips = " ".join(
                f'<span style="display:inline-block; background:{color}22;'
                f' border:1px solid {color}; color:#e2e8f0; padding:0.05em 0.5em;'
                f' border-radius:999px; font-size:0.8em; margin:0.1em;">'
                f'{_html.escape(term)} · {n}</span>'
                for term, n in cell.top_terms
            )
            terms_lines.append(
                f'<div style="margin-top:0.3em;">'
                f'<span style="color:{color}; font-size:0.85em; font-weight:600; '
                f'margin-right:0.4em;">{sent}</span>{chips}</div>'
            )
        rows.append(f'''
        <div style="border:1px solid #334155; border-radius:6px; padding:0.8em 1em;
                    margin-bottom:0.6em; background:#0f172a; color:#e2e8f0;">
          <div style="display:flex; justify-content:space-between; margin-bottom:0.4em;">
            <div style="font-weight:600;">
              {_html.escape(ASPECT_LABEL_VI.get(asp.aspect_category, asp.aspect_category))}
              <span style="color:#94a3b8; font-size:0.85em; margin-left:0.4em;">
                ({_html.escape(asp.aspect_category)})
              </span>
            </div>
            <div style="font-size:0.85em; color:#94a3b8;">
              {asp.total_mentions} mentions · {asp.n_reviews} reviews
            </div>
          </div>
          <div style="display:flex; height:8px; border-radius:4px; overflow:hidden;">
            <div style="width:{pos_pct:.1f}%; background:{SENT_COLOR['positive']};"></div>
            <div style="width:{neg_pct:.1f}%; background:{SENT_COLOR['negative']};"></div>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:0.75em;
                      color:#94a3b8; margin-top:0.2em;">
            <span>{bar_pos} positive</span>
            <span>{bar_neg} negative</span>
          </div>
          {''.join(terms_lines)}
        </div>
        ''')
    return "".join(rows)


def _render_actions(actions) -> str:
    if not actions:
        return (
            '<div style="padding:1em; color:#94a3b8; font-style:italic;">'
            "Không có hành động đề xuất.</div>"
        )
    cards = []
    priority_order = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(actions, key=lambda a: priority_order.get(a.priority, 99))
    for a in ordered:
        pcolor = PRIORITY_COLOR.get(a.priority, "#64748b")
        scolor = SENT_COLOR.get(a.sentiment, "#94a3b8")
        ev = ", ".join(_html.escape(t) for t in a.evidence_terms) or "—"
        cards.append(f'''
        <div style="border:1px solid #334155; border-left:4px solid {pcolor};
                    border-radius:6px; padding:0.8em 1em; margin-bottom:0.6em;
                    background:#0f172a; color:#e2e8f0;">
          <div style="display:flex; gap:0.4em; align-items:center; margin-bottom:0.4em;">
            <span style="background:{pcolor}; color:#0f172a; padding:0.1em 0.6em;
                         border-radius:999px; font-weight:700; font-size:0.75em;
                         text-transform:uppercase;">{_html.escape(a.priority)}</span>
            <span style="background:{scolor}22; color:{scolor}; padding:0.1em 0.6em;
                         border-radius:999px; font-size:0.8em;">
              {_html.escape(a.sentiment)}
            </span>
            <span style="color:#94a3b8; font-size:0.85em;">
              {_html.escape(ASPECT_LABEL_VI.get(a.aspect_category, a.aspect_category))}
            </span>
          </div>
          <div style="font-size:1.1em; font-weight:600; color:#fef3c7; line-height:1.4;">
            {_html.escape(a.action)}
          </div>
          <div style="margin-top:0.4em; color:#94a3b8; font-size:0.78em;">
            Evidence: {ev}
          </div>
        </div>
        ''')
    return "".join(cards)


def build_app(pipeline: CausaSentPipeline, use_gemini: bool = True) -> gr.Blocks:
    def infer_single(review: str):
        review = review.strip()
        if not review:
            return (
                '<div style="padding:1em; color:#94a3b8;">Nhập review tiếng Việt vào ô phía trên.</div>',
                "",
                "",
            )
        tuples = pipeline(review)
        status = (
            f"**{len(tuples)} aspect tuple** trích xuất được."
            if tuples
            else "**Không tìm thấy aspect.** Review quá ngắn hoặc mơ hồ."
        )
        return _highlight_review(review, tuples), _render_tuple_cards(tuples), status

    def infer_batch(raw_text: str):
        reviews = [r.strip() for r in raw_text.splitlines() if r.strip()]
        if not reviews:
            empty = (
                '<div style="padding:1em; color:#94a3b8;">'
                "Dán nhiều review (mỗi dòng 1 review) hoặc bấm Ví dụ.</div>"
            )
            return empty, empty, ""
        per_review_tuples = [pipeline(r) for r in reviews]
        summary = aggregate(per_review_tuples)
        actions = (
            generate_actions(summary) if use_gemini else template_actions(summary)
        )
        status = (
            f"**{summary.n_reviews} reviews** · {summary.n_tuples} tuples · "
            f"{len(actions)} action recommendations"
        )
        return _render_summary(summary), _render_actions(actions), status

    with gr.Blocks(
        title="CausaSent — Vietnamese Review Analyzer",
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
        css=".gradio-container {max-width: 1100px !important; margin: 0 auto !important;}",
    ) as app:
        gr.Markdown(
            "# 🍊 CausaSent v2 — ABSA + Action Recommendations\n"
            "Trích xuất `(aspect_term, aspect_category, sentiment)` từ review "
            "tiếng Việt; tổng hợp N reviews thành action recommendations cho người bán.\n\n"
            "*PhoBERT-large (ATE BIO + binary sentiment + supervised contrastive) · "
            "Gemini 2.5 Flash for action generation*"
        )

        with gr.Tabs():
            with gr.Tab("Một review"):
                inp = gr.Textbox(
                    label="Review tiếng Việt",
                    lines=5,
                    placeholder="Ship lâu nhưng đóng gói đẹp...",
                )
                btn = gr.Button("Phân tích", variant="primary", size="lg")
                gr.Examples(examples=EXAMPLES_SINGLE, inputs=inp, label="Ví dụ")
                status_s = gr.Markdown()
                gr.Markdown("### Review (aspect_term tô màu theo sentiment)")
                hl = gr.HTML()
                gr.Markdown("### Aspect tuples")
                cards = gr.HTML()
                btn.click(infer_single, inputs=inp, outputs=[hl, cards, status_s])
                inp.submit(infer_single, inputs=inp, outputs=[hl, cards, status_s])

            with gr.Tab("Nhiều review (batch)"):
                batch_inp = gr.Textbox(
                    label="Mỗi dòng 1 review",
                    lines=10,
                    placeholder="Review 1...\nReview 2...\nReview 3...",
                )
                btn_b = gr.Button("Tổng hợp + đề xuất hành động", variant="primary", size="lg")
                gr.Examples(examples=[[EXAMPLES_BATCH]], inputs=batch_inp, label="Ví dụ")
                status_b = gr.Markdown()
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Tổng hợp aspect")
                        summary_html = gr.HTML()
                    with gr.Column():
                        gr.Markdown("### Hành động đề xuất")
                        actions_html = gr.HTML()
                btn_b.click(
                    infer_batch,
                    inputs=batch_inp,
                    outputs=[summary_html, actions_html, status_b],
                )

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phobert-ckpt",
        default=os.environ.get("PHOBERT_CKPT", "checkpoints/phobert/best.pt"),
    )
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--no-gemini", action="store_true", help="use template actions only")
    args = ap.parse_args()

    pipeline = CausaSentPipeline(phobert_ckpt=args.phobert_ckpt)
    build_app(pipeline, use_gemini=not args.no_gemini).launch(share=args.share)


if __name__ == "__main__":
    main()

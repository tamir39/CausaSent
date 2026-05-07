"""Gradio demo that runs the full pipeline on a single review."""
from __future__ import annotations

import argparse
import html as _html
import os

import gradio as gr

from ..inference.pipeline import CausaSentPipeline


SENT_COLOR = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral":  "#a3a3a3",
}
ASPECT_LABEL_VI = {
    "delivery":          "Giao hàng",
    "packaging":         "Đóng gói",
    "product_quality":   "Chất lượng SP",
    "price":             "Giá",
    "customer_service":  "Hỗ trợ KH",
    "usability":         "Trải nghiệm",
    "appearance":        "Hình thức",
}

EXAMPLES = [
    "Ship lâu nhưng đóng gói đẹp, sản phẩm dùng tạm được, giá cũng ổn.",
    "Áo chất vải mịn, mặc thoáng, shop tư vấn nhiệt tình, giao nhanh.",
    "Hàng giao chậm 5 ngày, hộp móp méo, nhân viên trả lời cộc lốc, thất vọng.",
    "Giá hơi cao so với chất lượng, nhưng màu sắc và kiểu dáng thì ưng.",
]


def _highlight_review(review: str, results) -> str:
    """Render review HTML with cause spans underlined per sentiment color."""
    if not results:
        return f'<div style="padding:0.6em 0.9em; line-height:1.6; font-size:1.05em; background:#1e293b; border-radius:6px; color:#e2e8f0;">{_html.escape(review)}</div>'
    spans = sorted(
        ((r.cause_span[0], r.cause_span[1], r.sentiment) for r in results if r.cause_span),
        key=lambda x: x[0],
    )
    out, cur = [], 0
    for s, e, sent in spans:
        if s < cur:
            continue
        out.append(_html.escape(review[cur:s]))
        color = SENT_COLOR.get(sent, "#fff")
        out.append(
            f'<span style="border-bottom:3px solid {color}; padding:0 2px; '
            f'background:{color}22;">{_html.escape(review[s:e])}</span>'
        )
        cur = e
    out.append(_html.escape(review[cur:]))
    return (
        '<div style="padding:0.8em 1em; line-height:1.8; font-size:1.05em; '
        'background:#1e293b; border-radius:6px; color:#e2e8f0;">'
        + "".join(out)
        + "</div>"
    )


def _render_cards(results) -> str:
    """Render each tuple as a card so long causes/actions don't get truncated."""
    if not results:
        return (
            '<div style="padding:1em; color:#94a3b8; font-style:italic;">'
            "Chưa trích xuất được aspect nào. Thử review dài hơn hoặc rõ ràng hơn."
            "</div>"
        )
    cards = []
    for i, r in enumerate(results, 1):
        color = SENT_COLOR.get(r.sentiment, "#a3a3a3")
        aspect_vi = ASPECT_LABEL_VI.get(r.aspect, r.aspect)
        cards.append(f'''
        <div style="border:1px solid #334155; border-left:4px solid {color};
                    border-radius:6px; padding:0.8em 1em; margin-bottom:0.7em;
                    background:#0f172a; color:#e2e8f0;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4em;">
            <div>
              <span style="font-weight:600; font-size:1.05em;">#{i} · {_html.escape(aspect_vi)}</span>
              <span style="color:#94a3b8; font-size:0.9em; margin-left:0.4em;">({_html.escape(r.aspect)})</span>
            </div>
            <span style="background:{color}; color:#0f172a; padding:0.15em 0.65em;
                         border-radius:999px; font-weight:600; font-size:0.85em;">
              {_html.escape(r.sentiment)}
            </span>
          </div>
          <div style="margin:0.3em 0;">
            <span style="color:#94a3b8; font-size:0.85em;">Nguyên nhân:</span>
            <span style="border-bottom:2px solid {color}; padding:0 2px;">{_html.escape(r.cause_text)}</span>
            <span style="color:#64748b; font-size:0.8em; margin-left:0.4em;">[{r.cause_span[0]}, {r.cause_span[1]})</span>
          </div>
          <div style="margin-top:0.5em;">
            <span style="color:#94a3b8; font-size:0.85em;">Hành động:</span>
            <span style="font-weight:500;">{_html.escape(r.action)}</span>
          </div>
          <div style="margin-top:0.4em; color:#64748b; font-size:0.8em;">
            confidence: {r.confidence:.2f}
          </div>
        </div>
        ''')
    return "".join(cards)


def build_app(pipeline: CausaSentPipeline) -> gr.Blocks:
    def infer(review: str):
        review = review.strip()
        if not review:
            return (
                '<div style="padding:1em; color:#94a3b8;">Nhập review tiếng Việt vào ô phía trên.</div>',
                "",
                "",
            )
        results = pipeline(review)
        highlighted = _highlight_review(review, results)
        cards = _render_cards(results)
        status = (
            f"**{len(results)} bộ ba** trích xuất được."
            if results
            else "**Không tìm thấy aspect.** Mô hình baseline F1 ≈ 0,34 trên test — review ngắn / mơ hồ thường bỏ sót."
        )
        return highlighted, cards, status

    with gr.Blocks(
        title="CausaSent — Vietnamese Review Analyzer",
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
        css=".gradio-container {max-width: 1100px !important; margin: 0 auto !important;}",
    ) as app:
        gr.Markdown(
            "# 🍊 CausaSent\n"
            "Trích xuất bộ bốn `(khía cạnh, sắc thái, nguyên nhân, hành động)` "
            "từ review thương mại điện tử tiếng Việt.\n\n"
            "*PhoBERT-large + mT5-base · gold F1 0.34 (aspect-sent) / 0.46 (cause) · "
            "[dataset](https://huggingface.co/datasets/Tamir39/causasent) · "
            "[mô hình](https://huggingface.co/Tamir39/causasent-phobert)*"
        )

        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Textbox(
                    label="Review tiếng Việt",
                    lines=5,
                    placeholder="Ship lâu nhưng đóng gói đẹp, giá cũng ổn...",
                )
                btn = gr.Button("Phân tích", variant="primary", size="lg")
                gr.Examples(examples=EXAMPLES, inputs=inp, label="Ví dụ")

        status = gr.Markdown()
        gr.Markdown("### Review (cause spans được tô màu theo sắc thái)")
        highlighted = gr.HTML()
        gr.Markdown("### Các bộ ba trích xuất")
        cards = gr.HTML()

        btn.click(infer, inputs=inp, outputs=[highlighted, cards, status])
        inp.submit(infer, inputs=inp, outputs=[highlighted, cards, status])

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phobert-ckpt", default=os.environ.get("PHOBERT_CKPT", "checkpoints/phobert/best.pt"))
    ap.add_argument("--mt5-ckpt", default=os.environ.get("MT5_CKPT", "checkpoints/mt5/best.pt"))
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    pipeline = CausaSentPipeline(
        phobert_ckpt=args.phobert_ckpt,
        mt5_ckpt=args.mt5_ckpt,
    )
    build_app(pipeline).launch(share=args.share)


if __name__ == "__main__":
    main()

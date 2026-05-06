"""Gradio demo that runs the full pipeline on a single review."""
from __future__ import annotations

import argparse
import os

import gradio as gr

from ..inference.pipeline import CausaSentPipeline


def build_app(pipeline: CausaSentPipeline) -> gr.Blocks:
    def infer(review: str):
        review = review.strip()
        if not review:
            return [], "Nhập review tiếng Việt vào ô bên trái."
        results = pipeline(review)
        if not results:
            return [], "Không tìm thấy aspect nào trong review."
        rows = [
            [r.aspect, r.sentiment, r.cause_text, f"[{r.cause_span[0]},{r.cause_span[1]})", r.action]
            for r in results
        ]
        return rows, f"{len(results)} tuple(s) trích xuất được."

    with gr.Blocks(title="CausaSent — Vietnamese Review Analyzer") as app:
        gr.Markdown("# CausaSent\nCausal & actionable sentiment analysis cho review tiếng Việt.")
        with gr.Row():
            inp = gr.Textbox(label="Review", lines=5, placeholder="Ship lâu nhưng đóng gói đẹp")
            with gr.Column():
                status = gr.Markdown()
                out = gr.Dataframe(
                    headers=["Aspect", "Sentiment", "Cause", "Span", "Action"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                )
        btn = gr.Button("Phân tích", variant="primary")
        btn.click(infer, inputs=inp, outputs=[out, status])
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

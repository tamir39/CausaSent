"""Lightweight Gradio UI for reviewing/correcting weak-labeled annotations.

Usage:
    python -m src.tools.review_ui --in data/processed/weak.json --out data/gold/refined.json

Workflow per review:
1. Pick the review id from the dropdown.
2. Inspect the rendered review with cause spans highlighted.
3. Edit the JSON of `annotations` directly (Gradio Code box).
4. Click **Save** — the validator runs and the row is written to --out.
   Drop reasons are surfaced in the status box.

The UI is intentionally simple — it relies on the validator
(`src.data.validate.validate_review`) to enforce schema rules and span
correctness, so the human only needs to edit text.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import gradio as gr

from ..data.schema import Review, load_reviews, save_reviews
from ..data.validate import ValidationStats, validate_review


def _load(in_path: str) -> list[Review]:
    p = Path(in_path)
    if not p.exists():
        return []
    return load_reviews(p)


def _save_existing(out_path: str) -> dict[str, Review]:
    p = Path(out_path)
    if not p.exists():
        return {}
    return {r.id: r for r in load_reviews(p)}


def _highlight(review_text: str, anns: list[dict]) -> str:
    """Render the review with cause spans wrapped in <mark>."""
    spans: list[tuple[int, int, str]] = []
    for a in anns:
        try:
            s, e = a["cause_span"]
            label = f"{a.get('aspect', '?')}/{a.get('sentiment', '?')}"
            spans.append((int(s), int(e), label))
        except Exception:
            continue
    spans.sort(key=lambda x: x[0])
    out: list[str] = []
    cursor = 0
    for s, e, label in spans:
        if s < cursor or e > len(review_text):
            continue
        out.append(html.escape(review_text[cursor:s]))
        out.append(
            f'<mark title="{html.escape(label)}" style="background:#fff3a3;'
            f'padding:1px 2px;border-radius:2px">'
            f"{html.escape(review_text[s:e])}</mark>"
        )
        cursor = e
    out.append(html.escape(review_text[cursor:]))
    return f'<div style="font-family:sans-serif;line-height:1.6">{"".join(out)}</div>'


def build_ui(in_path: str, out_path: str) -> gr.Blocks:
    reviews = _load(in_path)
    saved = _save_existing(out_path)
    ids = [r.id for r in reviews]
    by_id = {r.id: r for r in reviews}

    def _initial_anns(review_id: str) -> list[dict]:
        if review_id in saved:
            return [a.model_dump() for a in saved[review_id].annotations]
        return [a.model_dump() for a in by_id[review_id].annotations]

    def on_select(review_id: str):
        if review_id not in by_id:
            return "", "[]", ""
        r = by_id[review_id]
        anns = _initial_anns(review_id)
        return r.review, json.dumps(anns, ensure_ascii=False, indent=2), _highlight(r.review, anns)

    def on_render(review_id: str, anns_json: str):
        if review_id not in by_id:
            return "", ""
        r = by_id[review_id]
        try:
            anns = json.loads(anns_json) if anns_json.strip() else []
        except json.JSONDecodeError as e:
            return f"<pre style='color:red'>JSON error: {e}</pre>", "JSON parse failed"
        return _highlight(r.review, anns), "rendered"

    def on_save(review_id: str, anns_json: str):
        if review_id not in by_id:
            return "no review selected"
        r = by_id[review_id]
        try:
            raw_anns = json.loads(anns_json) if anns_json.strip() else []
        except json.JSONDecodeError as e:
            return f"JSON error: {e}"
        stats = ValidationStats()
        validated = validate_review(r.id, r.review, raw_anns, stats=stats)
        if validated is None:
            return f"validation rejected the review entirely. stats={stats.as_dict()}"
        saved[review_id] = validated
        save_reviews(list(saved.values()), out_path)
        return f"saved {review_id} ({len(validated.annotations)} ann). drops={stats.as_dict()}"

    with gr.Blocks(title="CausaSent annotation review") as demo:
        gr.Markdown(f"## CausaSent annotation review\n`in:` {in_path}  `out:` {out_path}")
        with gr.Row():
            id_dd = gr.Dropdown(choices=ids, value=ids[0] if ids else None, label="review id", scale=2)
            status = gr.Textbox(label="status", interactive=False, scale=3)
        review_box = gr.Textbox(label="review text (read-only)", interactive=False, lines=3)
        rendered = gr.HTML(label="highlighted spans")
        anns_box = gr.Code(label="annotations (edit JSON)", language="json", lines=15)
        with gr.Row():
            render_btn = gr.Button("Re-render highlights")
            save_btn = gr.Button("Save", variant="primary")

        id_dd.change(on_select, inputs=id_dd, outputs=[review_box, anns_box, rendered])
        render_btn.click(on_render, inputs=[id_dd, anns_box], outputs=[rendered, status])
        save_btn.click(on_save, inputs=[id_dd, anns_box], outputs=status)

        if ids:
            demo.load(on_select, inputs=id_dd, outputs=[review_box, anns_box, rendered])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="input JSON (weak-labeled)")
    ap.add_argument("--out", dest="out_path", required=True, help="output JSON (corrected)")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    build_ui(args.in_path, args.out_path).launch(share=args.share)


if __name__ == "__main__":
    main()

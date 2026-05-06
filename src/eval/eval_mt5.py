"""ROUGE-L for mT5 action generation, plus a CSV dump for human evaluation."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
import yaml
from rouge_score import rouge_scorer

from ..data.dataset import build_action_examples
from ..data.schema import load_reviews
from ..models.mt5_action import generate_action, load_mt5


def evaluate(ckpt_path: str, config_path: str, split: str = "test", n_human: int = 50) -> dict:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_mt5(cfg["model"]["pretrained"])
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()

    path_key = {"train": "train_path", "val": "val_path", "test": "test_path"}[split]
    reviews = load_reviews(cfg["data"][path_key])
    examples = build_action_examples(reviews)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_sum = 0.0
    rows: list[tuple[str, str, str]] = []
    for ex in examples:
        # Generate by feeding the structured prompt directly.
        enc = tokenizer(ex.input_text, return_tensors="pt", truncation=True, max_length=cfg["data"]["max_input_len"]).to(device)
        gen_kwargs = {
            "num_beams": cfg["generate"]["num_beams"],
            "max_new_tokens": cfg["generate"]["max_new_tokens"],
            "length_penalty": cfg["generate"]["length_penalty"],
        }
        out = model.generate(**enc, **gen_kwargs)
        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        rouge_sum += scorer.score(ex.target_text, pred)["rougeL"].fmeasure
        rows.append((ex.input_text, ex.target_text, pred))

    rouge_l = rouge_sum / max(1, len(examples))
    print(f"ROUGE-L (avg, {len(examples)} samples): {rouge_l:.4f}")

    # Sample for human eval.
    n = min(n_human, len(rows))
    sample = random.sample(rows, n)
    out_csv = Path(cfg["output"]["ckpt_dir"]).parent.parent / "outputs" / f"action_human_eval_{split}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input", "gold_action", "predicted_action", "is_reasonable_0_1", "is_actionable_0_1", "notes"])
        for inp, gold, pred in sample:
            w.writerow([inp, gold, pred, "", "", ""])
    print(f"Human-eval CSV → {out_csv}")
    return {"rouge_l": rouge_l, "n": len(examples)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n-human", type=int, default=50)
    args = ap.parse_args()
    evaluate(args.ckpt, args.config, args.split, args.n_human)


if __name__ == "__main__":
    main()

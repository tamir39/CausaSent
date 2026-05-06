"""mT5-base wrapper for action generation."""
from __future__ import annotations

from transformers import AutoTokenizer, MT5ForConditionalGeneration

from ..data.dataset import ACTION_INPUT_TEMPLATE


def load_mt5(pretrained: str = "google/mt5-base"):
    tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True)
    model = MT5ForConditionalGeneration.from_pretrained(pretrained)
    return model, tokenizer


def format_action_input(aspect: str, sentiment: str, cause: str, review: str) -> str:
    return ACTION_INPUT_TEMPLATE.format(
        aspect=aspect, sentiment=sentiment, cause=cause, review=review,
    )


def generate_action(
    model,
    tokenizer,
    aspect: str,
    sentiment: str,
    cause: str,
    review: str,
    *,
    num_beams: int = 4,
    max_new_tokens: int = 32,
    length_penalty: float = 1.0,
    device: str | None = None,
) -> str:
    text = format_action_input(aspect, sentiment, cause, review)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    if device:
        enc = {k: v.to(device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        length_penalty=length_penalty,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

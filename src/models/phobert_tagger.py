"""PhoBERT-large with two parallel token-classification heads.

Head A: aspect-sentiment BIO (43 labels)
Head B: cause BIO            (3 labels)

Both heads share the same encoder. Loss is the (weighted) sum of the two
per-head cross-entropies, with `IGNORE_INDEX` masking sub-word and special
positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ..data.label_schema import (
    ASPECT_SENT_LABELS,
    CAUSE_LABELS,
    IGNORE_INDEX,
)


@dataclass
class TaggerOutput:
    loss: torch.Tensor | None
    asp_logits: torch.Tensor    # (B, T, n_asp)
    cause_logits: torch.Tensor  # (B, T, n_cause)


class PhoBertTwoHeadTagger(nn.Module):
    def __init__(
        self,
        pretrained: str = "vinai/phobert-large",
        dropout: float = 0.1,
        asp_class_weights: torch.Tensor | None = None,
        cause_class_weights: torch.Tensor | None = None,
        cause_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.asp_head = nn.Linear(hidden, len(ASPECT_SENT_LABELS))
        self.cause_head = nn.Linear(hidden, len(CAUSE_LABELS))
        self.cause_loss_weight = cause_loss_weight

        self.asp_loss_fn = nn.CrossEntropyLoss(
            weight=asp_class_weights, ignore_index=IGNORE_INDEX
        )
        self.cause_loss_fn = nn.CrossEntropyLoss(
            weight=cause_class_weights, ignore_index=IGNORE_INDEX
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        asp_labels: torch.Tensor | None = None,
        cause_labels: torch.Tensor | None = None,
    ) -> TaggerOutput:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(out.last_hidden_state)
        asp_logits = self.asp_head(h)
        cause_logits = self.cause_head(h)

        loss: torch.Tensor | None = None
        if asp_labels is not None and cause_labels is not None:
            asp_loss = self.asp_loss_fn(
                asp_logits.view(-1, asp_logits.size(-1)), asp_labels.view(-1)
            )
            cause_loss = self.cause_loss_fn(
                cause_logits.view(-1, cause_logits.size(-1)), cause_labels.view(-1)
            )
            loss = asp_loss + self.cause_loss_weight * cause_loss
        return TaggerOutput(loss=loss, asp_logits=asp_logits, cause_logits=cause_logits)

    @classmethod
    def load_tokenizer(cls, pretrained: str = "vinai/phobert-large"):
        return AutoTokenizer.from_pretrained(pretrained, use_fast=True)

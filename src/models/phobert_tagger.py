"""PhoBERT-large with two parallel token-classification heads for ABSA.

Head A (ATE): aspect-term BIO + category (15 labels)
  - Per token: O | B-{aspect} | I-{aspect}
  - Loss: CrossEntropy over all token positions

Head B (Sentiment): binary sentiment classifier (2 labels)
  - Per token: positive (0) | negative (1)
  - Loss: CrossEntropy only at B-token positions (IGNORE_INDEX elsewhere)

Both heads share the same PhoBERT-large encoder.
Optional: contrastive loss on B-token embeddings grouped by aspect_category.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ..data.label_schema import (
    ATE_LABELS,
    IGNORE_INDEX,
    SENTIMENT_LABELS,
)


@dataclass
class TaggerOutput:
    loss: torch.Tensor | None
    ate_logits: torch.Tensor   # (B, T, 15)
    sent_logits: torch.Tensor  # (B, T, 2)


class PhoBertABSA(nn.Module):
    def __init__(
        self,
        pretrained: str = "vinai/phobert-large",
        dropout: float = 0.1,
        ate_class_weights: torch.Tensor | None = None,
        sent_class_weights: torch.Tensor | None = None,
        sent_loss_weight: float = 1.0,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.07,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.ate_head = nn.Linear(hidden, len(ATE_LABELS))
        self.sent_head = nn.Linear(hidden, len(SENTIMENT_LABELS))
        self.sent_loss_weight = sent_loss_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature

        self.ate_loss_fn = nn.CrossEntropyLoss(
            weight=ate_class_weights, ignore_index=IGNORE_INDEX
        )
        self.sent_loss_fn = nn.CrossEntropyLoss(
            weight=sent_class_weights, ignore_index=IGNORE_INDEX
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ate_labels: torch.Tensor | None = None,
        sent_labels: torch.Tensor | None = None,
    ) -> TaggerOutput:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(out.last_hidden_state)
        ate_logits = self.ate_head(h)
        sent_logits = self.sent_head(h)

        loss: torch.Tensor | None = None
        if ate_labels is not None and sent_labels is not None:
            ate_loss = self.ate_loss_fn(
                ate_logits.view(-1, ate_logits.size(-1)), ate_labels.view(-1)
            )
            sent_loss = self.sent_loss_fn(
                sent_logits.view(-1, sent_logits.size(-1)), sent_labels.view(-1)
            )
            loss = ate_loss + self.sent_loss_weight * sent_loss

            if self.contrastive_weight > 0 and ate_labels is not None:
                cl = self._contrastive_loss(h, ate_labels)
                if cl is not None:
                    loss = loss + self.contrastive_weight * cl

        return TaggerOutput(loss=loss, ate_logits=ate_logits, sent_logits=sent_logits)

    def _contrastive_loss(
        self, h: torch.Tensor, ate_labels: torch.Tensor
    ) -> torch.Tensor | None:
        """Supervised contrastive loss on B-token embeddings grouped by aspect_category.

        Pulls embeddings of the same aspect_category together and pushes different
        categories apart. Only B-token positions are used (not I-tokens or O).
        Returns None if fewer than 2 valid B-tokens are found in the batch.
        """
        from ..data.label_schema import ATE_ID2LABEL, ASPECTS, parse_ate_tag

        B, T, _ = h.shape
        embeds: list[torch.Tensor] = []
        labels: list[int] = []

        for b in range(B):
            for t in range(T):
                lbl = ate_labels[b, t].item()
                if lbl == IGNORE_INDEX or lbl == 0:  # 0 = O label
                    continue
                tag = ATE_ID2LABEL.get(int(lbl), "O")
                parsed = parse_ate_tag(tag)
                if parsed is None or parsed[0] != "B":
                    continue
                _, asp = parsed
                asp_idx = list(ASPECTS).index(asp)
                embeds.append(F.normalize(h[b, t], dim=0))
                labels.append(asp_idx)

        if len(embeds) < 2:
            return None

        emb_mat = torch.stack(embeds)                           # (N, D)
        lbl_tensor = torch.tensor(labels, device=h.device)      # (N,)
        sim = torch.matmul(emb_mat, emb_mat.T) / self.contrastive_temperature  # (N, N)

        N = len(embeds)
        # Mask out self-similarity on the diagonal
        mask_self = ~torch.eye(N, dtype=torch.bool, device=h.device)
        loss_sum = torch.tensor(0.0, device=h.device)
        count = 0
        for i in range(N):
            pos_mask = (lbl_tensor == lbl_tensor[i]) & mask_self[i]
            if not pos_mask.any():
                continue
            # log-sum-exp over all non-self pairs as denominator
            neg_sim = sim[i][mask_self[i]]
            log_denom = torch.logsumexp(neg_sim, dim=0)
            loss_sum = loss_sum + (-sim[i][pos_mask] + log_denom).mean()
            count += 1

        return loss_sum / count if count > 0 else None

    @classmethod
    def load_tokenizer(cls, pretrained: str = "vinai/phobert-large"):
        return AutoTokenizer.from_pretrained(pretrained, use_fast=True)


# Backward-compatible alias for code that still references the old class name.
PhoBertTwoHeadTagger = PhoBertABSA

"""End-to-end inference: review text → list of (aspect, sentiment, cause, action)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from ..data.label_schema import IGNORE_INDEX
from ..data.segmenter import get_segmenter
from ..data.span_align import word_boundaries
from ..models.mt5_action import generate_action, load_mt5
from ..models.phobert_tagger import PhoBertTwoHeadTagger
from .decode import ExtractedTuple, decode_to_tuples


@dataclass
class PredictedTuple:
    aspect: str
    sentiment: str
    cause_text: str
    cause_span: tuple[int, int]
    action: str
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class CausaSentPipeline:
    def __init__(
        self,
        phobert_ckpt: str | Path,
        mt5_ckpt: str | Path,
        phobert_pretrained: str = "vinai/phobert-large",
        mt5_pretrained: str = "google/mt5-base",
        segmenter_kind: str = "vncorenlp",
        device: str | None = None,
        min_confidence: float = 0.0,
    ):
        self.min_confidence = min_confidence
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tagger_tok = PhoBertTwoHeadTagger.load_tokenizer(phobert_pretrained)
        self.tagger = PhoBertTwoHeadTagger(pretrained=phobert_pretrained)
        state = torch.load(phobert_ckpt, map_location=self.device)
        self.tagger.load_state_dict(state["model"] if "model" in state else state)
        self.tagger.to(self.device).eval()

        self.mt5, self.mt5_tok = load_mt5(mt5_pretrained)
        mt5_state = torch.load(mt5_ckpt, map_location=self.device)
        self.mt5.load_state_dict(mt5_state["model"] if "model" in mt5_state else mt5_state)
        self.mt5.to(self.device).eval()

        self.segmenter = get_segmenter(segmenter_kind)

    @torch.no_grad()
    def _tag(self, review: str) -> tuple[list[ExtractedTuple], list[tuple[int, int]]]:
        words = self.segmenter.segment(review)
        if not words:
            return [], []
        enc = self.tagger_tok(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        out = self.tagger(enc["input_ids"], enc["attention_mask"])
        asp_logits = out.asp_logits[0]
        cau_logits = out.cause_logits[0]
        asp_pred = asp_logits.argmax(-1).tolist()
        cau_pred = cau_logits.argmax(-1).tolist()
        asp_probs_sub = torch.softmax(asp_logits, dim=-1).tolist()

        # Reduce subword preds → word-level (take the first subword per word).
        word_ids = enc.word_ids(batch_index=0)
        n_words = len(words)
        asp_word = [0] * n_words
        cau_word = [0] * n_words
        asp_probs_word: list[list[float]] = [[1.0]] * n_words
        seen: set[int] = set()
        for i, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            asp_word[wid] = asp_pred[i]
            cau_word[wid] = cau_pred[i]
            asp_probs_word[wid] = asp_probs_sub[i]

        w_spans = word_boundaries(review, words)
        if len(w_spans) < n_words:
            n_words = len(w_spans)
            asp_word = asp_word[:n_words]
            cau_word = cau_word[:n_words]
            asp_probs_word = asp_probs_word[:n_words]
        tuples = decode_to_tuples(
            asp_word, cau_word, w_spans, review,
            asp_probs=asp_probs_word,
            min_confidence=self.min_confidence,
        )
        return tuples, w_spans

    def __call__(self, review: str) -> list[PredictedTuple]:
        extracted, _ = self._tag(review)
        results: list[PredictedTuple] = []
        for t in extracted:
            action = generate_action(
                self.mt5, self.mt5_tok,
                aspect=t.aspect, sentiment=t.sentiment, cause=t.cause_text, review=review,
                device=self.device,
            )
            results.append(PredictedTuple(
                aspect=t.aspect,
                sentiment=t.sentiment,
                cause_text=t.cause_text,
                cause_span=t.cause_span,
                action=action,
                confidence=t.confidence,
            ))
        return results


_ = IGNORE_INDEX  # silence unused import (kept for potential future masking use)

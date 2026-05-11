"""End-to-end inference: review text -> list of ABSA tuples.

Output per review: [(aspect_category, aspect_term, aspect_term_span, sentiment)]

Action generation is NOT per-tuple here. Call the aggregator separately:
  from .aggregator import aggregate_actions
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..data.label_schema import IGNORE_INDEX
from ..data.segmenter import get_segmenter
from ..data.span_align import word_boundaries
from ..models.phobert_tagger import PhoBertABSA
from .decode import ExtractedTuple, decode_to_tuples


@dataclass
class PredictedTuple:
    aspect_category: str
    aspect_term: str
    aspect_term_span: tuple[int, int]
    sentiment: str
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class CausaSentPipeline:
    def __init__(
        self,
        phobert_ckpt: str | Path,
        phobert_pretrained: str = "vinai/phobert-large",
        segmenter_kind: str = "vncorenlp",
        device: str | None = None,
        min_confidence: float = 0.0,
    ):
        self.min_confidence = min_confidence
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tagger_tok = PhoBertABSA.load_tokenizer(phobert_pretrained)
        self.tagger = PhoBertABSA(pretrained=phobert_pretrained)
        state = torch.load(phobert_ckpt, map_location=self.device)
        self.tagger.load_state_dict(
            state["model"] if "model" in state else state, strict=False
        )
        self.tagger.to(self.device).eval()
        self.segmenter = get_segmenter(segmenter_kind)

    @torch.no_grad()
    def _tag(self, review: str) -> list[ExtractedTuple]:
        words = self.segmenter.segment(review)
        if not words:
            return []
        enc = self.tagger_tok(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        out = self.tagger(enc["input_ids"], enc["attention_mask"])
        ate_logits = out.ate_logits[0]
        sent_logits = out.sent_logits[0]
        ate_pred = ate_logits.argmax(-1).tolist()
        sent_pred = sent_logits.argmax(-1).tolist()
        ate_probs = torch.softmax(ate_logits, dim=-1).tolist()

        try:
            word_ids = enc.word_ids(batch_index=0)
        except (ValueError, AttributeError):
            from ..data.dataset import _word_ids_slow
            seq_len = len(ate_pred)
            word_ids = _word_ids_slow(self.tagger_tok, words, max_len=seq_len)[:seq_len]

        n_words = len(words)
        ate_word = [0] * n_words
        sent_word = [0] * n_words
        ate_probs_word: list[list[float]] = [[1.0]] * n_words
        seen: set[int] = set()
        for i, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            ate_word[wid] = ate_pred[i]
            sent_word[wid] = sent_pred[i]
            ate_probs_word[wid] = ate_probs[i]

        w_spans = word_boundaries(review, words)
        if len(w_spans) < n_words:
            n_words = len(w_spans)
            ate_word = ate_word[:n_words]
            sent_word = sent_word[:n_words]
            ate_probs_word = ate_probs_word[:n_words]

        return decode_to_tuples(
            ate_word, sent_word, w_spans, review,
            ate_probs=ate_probs_word,
            min_confidence=self.min_confidence,
        )

    def __call__(self, review: str) -> list[PredictedTuple]:
        extracted = self._tag(review)
        return [
            PredictedTuple(
                aspect_category=t.aspect_category,
                aspect_term=t.aspect_term,
                aspect_term_span=t.aspect_term_span,
                sentiment=t.sentiment,
                confidence=t.confidence,
            )
            for t in extracted
        ]


_ = IGNORE_INDEX  # silence unused import

"""Thin wrapper around VnCoreNLP word segmentation.

VnCoreNLP needs a one-time download of the JAR + models. We lazy-load on first
use so importing this module is cheap. Set CAUSASENT_VNCORENLP_DIR to override
the install location (default: ./vncorenlp).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


class Segmenter(Protocol):
    def segment(self, text: str) -> list[str]: ...


class WhitespaceSegmenter:
    """Fallback segmenter — splits on whitespace. Use only when VnCoreNLP is unavailable."""

    def segment(self, text: str) -> list[str]:
        return text.split()


class VnCoreNLPSegmenter:
    _instance: "VnCoreNLPSegmenter | None" = None

    def __init__(self, save_dir: str | None = None):
        import py_vncorenlp  # lazy import

        save_dir = save_dir or os.environ.get(
            "CAUSASENT_VNCORENLP_DIR", str(Path.cwd() / "vncorenlp")
        )
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        if not any(Path(save_dir).glob("VnCoreNLP-*.jar")):
            py_vncorenlp.download_model(save_dir=save_dir)
        # `annotators=["wseg"]` keeps it lightweight.
        self._model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)

    def segment(self, text: str) -> list[str]:
        # py_vncorenlp returns sentences of underscore-joined words.
        sentences = self._model.word_segment(text)
        words: list[str] = []
        for s in sentences:
            words.extend(s.split())
        return words


def get_segmenter(kind: str = "vncorenlp") -> Segmenter:
    if kind == "none" or kind == "whitespace":
        return WhitespaceSegmenter()
    if kind == "vncorenlp":
        if VnCoreNLPSegmenter._instance is None:
            VnCoreNLPSegmenter._instance = VnCoreNLPSegmenter()
        return VnCoreNLPSegmenter._instance
    raise ValueError(f"unknown segmenter kind: {kind}")

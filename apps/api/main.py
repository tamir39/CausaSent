"""FastAPI inference service for the CausaSent PWA.

Endpoints:
  POST /analyze        — JSON {reviews: [str], use_llm?: bool, top_k?: int}
                          → {summary, actions, per_review}
  POST /analyze-csv    — multipart CSV upload + form fields
                          → same as /analyze
  POST /analyze-single — JSON {review: str} → {tuples: [...]}
  GET  /health         — {status, model_loaded}
  GET  /labels         — taxonomy info (aspect list, sentiments, VI labels)

Run:
    uvicorn apps.api.main:app --reload --port 8000

Env vars:
    PHOBERT_CKPT       path to best.pt (default: checkpoints/phobert/best.pt)
    GEMINI_API_KEY     required for use_llm=true (falls back to templates otherwise)
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
from typing import Iterator, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.data.label_schema import ASPECTS, SENTIMENTS
from src.inference.action_llm import (
    ASPECT_LABEL_VI,
    generate_actions,
    template_actions,
)
from src.inference.aggregate import aggregate
from src.inference.pipeline import CausaSentPipeline


# ---- Lazy model load (so the API can boot without weights for /health) -------

_PIPELINE: CausaSentPipeline | None = None


def _ckpt_path() -> str:
    return os.environ.get("PHOBERT_CKPT", "checkpoints/phobert/best.pt")


def _get_pipeline() -> CausaSentPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        ckpt = _ckpt_path()
        if not os.path.exists(ckpt):
            raise HTTPException(
                status_code=503,
                detail=f"Model checkpoint not found at {ckpt}. "
                       f"Set PHOBERT_CKPT env or download from HF.",
            )
        _PIPELINE = CausaSentPipeline(phobert_ckpt=ckpt)
    return _PIPELINE


# ---- Pydantic schemas --------------------------------------------------------

class AnalyzeRequest(BaseModel):
    # Cap matches the frontend MAX_REVIEWS so an oversized POST is rejected
    # at the boundary instead of OOMing the GPU or burning LLM quota.
    reviews: list[str] = Field(..., min_length=1, max_length=200)
    use_llm: bool = True
    top_k: int = 5
    min_confidence: float = 0.0


class SingleRequest(BaseModel):
    review: str


# ---- App ---------------------------------------------------------------------

app = FastAPI(
    title="CausaSent ABSA API",
    description="Vietnamese aspect-based sentiment analysis + action recommendations",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _PIPELINE is not None,
        "ckpt_exists": os.path.exists(_ckpt_path()),
        "gemini_enabled": bool(os.environ.get("GEMINI_API_KEY")),
    }


@app.get("/labels")
def labels() -> dict:
    return {
        "aspects": list(ASPECTS),
        "aspect_labels_vi": ASPECT_LABEL_VI,
        "sentiments": list(SENTIMENTS),
    }


@app.post("/analyze-single")
def analyze_single(req: SingleRequest) -> dict:
    pipeline = _get_pipeline()
    text = req.review.strip()
    if not text:
        raise HTTPException(status_code=400, detail="review is empty")
    tuples = pipeline(text)
    return {"review": text, "tuples": [t.to_dict() for t in tuples]}


def _analyze_core(
    reviews: list[str],
    *,
    use_llm: bool,
    top_k: int,
    min_confidence: float,
) -> dict:
    pipeline = _get_pipeline()
    per_review_records: list[dict] = []
    tuples_stream: list[list] = []
    for i, text in enumerate(reviews):
        t = (text or "").strip()
        if not t:
            tuples_stream.append([])
            per_review_records.append({"id": str(i), "review": "", "tuples": []})
            continue
        tuples = pipeline(t)
        tuples_stream.append(tuples)
        per_review_records.append({
            "id": str(i),
            "review": t,
            "tuples": [pt.to_dict() for pt in tuples],
        })
    summary = aggregate(tuples_stream, top_k=top_k, min_confidence=min_confidence)
    actions = generate_actions(summary) if use_llm else template_actions(summary)
    return {
        "n_reviews": len(reviews),
        "summary": summary.to_dict(),
        "actions": [a.to_dict() for a in actions],
        "per_review": per_review_records,
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    return _analyze_core(
        req.reviews,
        use_llm=req.use_llm,
        top_k=req.top_k,
        min_confidence=req.min_confidence,
    )


def _sse(event: str, data: dict) -> str:
    """Serialize one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_gen(req: "AnalyzeRequest") -> Iterator[str]:
    """Generator yielding SSE frames as the pipeline progresses.

    Frame sequence:
        event: start    → {n_reviews}
        event: review   → {idx, id, review, tuples}  (one per review)
        event: summary  → {summary}                  (after each review)
        event: actions_pending → {}
        event: actions  → {actions}
        event: done     → {n_tuples, n_actions, elapsed_ms}
    """
    started = time.monotonic()
    pipeline = _get_pipeline()

    yield _sse("start", {
        "n_reviews": len(req.reviews),
        "use_llm": req.use_llm,
    })

    per_review_tuples: list[list] = []

    for i, text in enumerate(req.reviews):
        t = (text or "").strip()
        tuples = pipeline(t) if t else []
        per_review_tuples.append(tuples)

        yield _sse("review", {
            "idx": i,
            "id": str(i),
            "review": t,
            "tuples": [pt.to_dict() for pt in tuples],
        })

        # Recompute summary so the dashboard charts update live.
        summary = aggregate(
            per_review_tuples, top_k=req.top_k, min_confidence=req.min_confidence
        )
        yield _sse("summary", {"summary": summary.to_dict()})

    yield _sse("actions_pending", {"use_llm": req.use_llm})
    actions = generate_actions(summary) if req.use_llm else template_actions(summary)
    yield _sse("actions", {"actions": [a.to_dict() for a in actions]})

    yield _sse("done", {
        "n_reviews": len(req.reviews),
        "n_tuples": summary.n_tuples,
        "n_actions": len(actions),
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    })


@app.post("/analyze-stream")
def analyze_stream(req: AnalyzeRequest):
    """Server-Sent-Events variant of /analyze for progressive UI updates."""
    return StreamingResponse(
        _stream_gen(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",      # disable nginx-style buffering
            "Connection": "keep-alive",
        },
    )


@app.post("/analyze-csv")
async def analyze_csv(
    file: UploadFile = File(...),
    review_col: Optional[str] = Form(None),
    use_llm: bool = Form(True),
    top_k: int = Form(5),
    min_confidence: float = Form(0.0),
) -> dict:
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    headers = reader.fieldnames or []
    col = review_col or _guess_review_column(headers)
    if not col:
        raise HTTPException(
            status_code=400,
            detail=f"Could not detect review column. Headers: {headers}. "
                   f"Pass review_col explicitly.",
        )
    reviews = [(row.get(col) or "").strip() for row in reader]
    reviews = [r for r in reviews if r]
    if not reviews:
        raise HTTPException(status_code=400, detail="No non-empty reviews found.")
    return _analyze_core(
        reviews,
        use_llm=use_llm,
        top_k=top_k,
        min_confidence=min_confidence,
    )


def _guess_review_column(headers: list[str]) -> str | None:
    candidates = ("review", "content", "comment", "text", "review_text")
    lower = {h.lower(): h for h in headers}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None

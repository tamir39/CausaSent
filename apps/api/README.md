# CausaSent FastAPI backend

Vietnamese ABSA inference service that exposes the `PhoBertABSA` checkpoint +
aggregator + Gemini action generator over HTTP for the PWA frontend.

## Endpoints

| Method | Path             | Purpose                                                  |
| ------ | ---------------- | -------------------------------------------------------- |
| GET    | `/health`        | Liveness + whether the checkpoint and Gemini key are set |
| GET    | `/labels`        | Aspect taxonomy + Vietnamese labels + sentiment set      |
| POST   | `/analyze-single`| One review → list of `(aspect_category, term, sentiment)`|
| POST   | `/analyze`       | JSON `{reviews: [str], use_llm?, top_k?}` → full summary |
| POST   | `/analyze-csv`   | Multipart CSV upload → full summary                      |

The full response shape is documented in `apps/web/lib/api.ts` (same types are
generated from `src/inference/aggregate.py` and `src/inference/action_llm.py`).

## Boot

```bash
# 1. install runtime deps (alongside the project requirements)
uv pip install -r apps/api/requirements.txt

# 2. point the service at a trained checkpoint
export PHOBERT_CKPT=checkpoints/phobert/best.pt
# optional: enable Gemini action generation
export GEMINI_API_KEY=...     # falls back to template actions without it

# 3. run
uvicorn apps.api.main:app --reload --port 8000
```

`/health` returns immediately even without a checkpoint (so containers can
boot before weights are downloaded). The model is lazy-loaded on the first
`/analyze*` call.

## Notes

- CORS is wide-open (`*`) — fine for a class project demo, lock down before
  any public deployment.
- The checkpoint path resolves relative to the working directory at startup.
  Run from the repo root so `checkpoints/phobert/best.pt` exists.
- `use_llm=true` requires `GEMINI_API_KEY` in env; otherwise the service
  silently falls back to deterministic Vietnamese templates from
  `src/inference/action_llm.py::template_actions`.

# CausaSent web (PWA)

Next.js (App Router) + Tailwind + Recharts client for the CausaSent FastAPI
backend.

## Pages

- `/`          — paste-reviews textarea or CSV upload, posts to `/analyze`.
- `/dashboard` — radar of aspect coverage, per-aspect sentiment breakdown,
                 prioritised action cards. Reads the last result from
                 `sessionStorage`.

## Boot

```bash
cd apps/web
npm install
# Point at the FastAPI backend (default: http://localhost:8000)
echo 'NEXT_PUBLIC_API_BASE=http://localhost:8000' > .env.local
npm run dev
```

## Notes

- This is a thin client. All inference + aggregation + LLM action generation
  happens in `apps/api`. The frontend only renders.
- Service worker (`public/sw.js`) caches the app shell so the page loads
  offline; `/analyze` requests always hit the network.
- Deploy: Vercel for the frontend; backend separately (the user's choice —
  Fly.io / Railway / a Kaggle session with `ngrok` for class demos).

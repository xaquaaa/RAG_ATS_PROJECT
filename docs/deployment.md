# Deployment

Two services: FastAPI backend on Render, Streamlit UI on Streamlit Community
Cloud. Both free tiers. Do the backend first — the frontend needs its URL.

## Prerequisites

- Supabase project set up per `docs/schema.sql` and
  `docs/schema_hybrid_search.sql` (see `docs/architecture.md` → Persistence
  for the pooler-URI note — use port 6543, not 5432).
- A Groq API key (console.groq.com).
- Code pushed to a GitHub repo Render/Streamlit Cloud can access.

## 1. Backend — Render

1. **New Web Service** → connect your GitHub repo.
2. **Build command:** `pip install -r requirements.txt`
3. **Start command:** `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
   (Render injects `$PORT` — don't hardcode 8000 here.)
4. **Environment variables** (Render dashboard → Environment): copy every
   key from your local `.env` — `GROQ_API_KEY`, `GROQ_MODEL`,
   `SUPABASE_DB_URL` (the **pooler** URI), `USE_SUPABASE=true`,
   `EVIDENCE_THRESHOLD`, `RRF_K`, `NEAR_MISS_MARGIN`, `NEAR_MISS_MAX`,
   `CONFIDENCE_HIGH_MARGIN`, `CONFIDENCE_MEDIUM_MARGIN`,
   `SHORTLIST_SIZE`, `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`. Never
   commit `.env` — set these in Render's dashboard only.
5. **Instance type:** free tier (512MB) works — but only because the
   embedding and cross-encoder models run on `fastembed` (ONNX runtime), not
   `sentence-transformers`/PyTorch. Two PyTorch-backed models plus PyTorch's
   own baseline overhead exceeded 512MB in initial testing. If you've
   modified `src/indexing/embeddings.py` or `src/retrieval/reranker.py` to
   reintroduce a PyTorch-based model, expect to hit this again. Also note:
   free Render services spin down after ~15 min idle and cold-start on the
   next request (30–50s) — expected, not a bug.
6. Deploy. Once live, confirm `https://<your-service>.onrender.com/health`
   responds — this also confirms Supabase connectivity, since `/health`
   queries the candidate count.
7. **Seed your dataset once** against the live URL — Render's free tier has
   **no shell access**, and its free-tier disk is ephemeral anyway (wiped on
   every redeploy/restart), so neither "commit resumes to git" nor "run the
   generator via shell" is a durable answer. Use the in-memory admin
   endpoint instead, which writes straight to Supabase without touching
   disk at all:
   ```bash
   # Temporarily set ADMIN_ENDPOINTS_ENABLED=true in Render's environment
   # variables first, redeploy, then:
   curl -X POST https://<your-service>.onrender.com/admin/seed-synthetic-data \
     -H "Content-Type: application/json" \
     -d '{"count": 100}'
   # Then set ADMIN_ENDPOINTS_ENABLED back to false (or remove it) and
   # redeploy again — this endpoint is unauthenticated and only meant for
   # one-off demo seeding with synthetic data, never left on.
   ```
   If you'd rather use your own real resume files instead of synthetic
   ones, `/ingest` (reading `data/resumes/`) still works — just note it
   requires those files to actually be present in the deployed repo, which
   means committing them (removing the `data/resumes/*.txt` line from
   `.gitignore`) since there's no way to get files onto a shell-less,
   ephemeral-disk free instance any other way.

## 2. Frontend — Streamlit Community Cloud

1. share.streamlit.io → **New app** → point at your repo,
   `streamlit_app/app.py` as the entrypoint.
2. **Secrets** (Streamlit Cloud dashboard → Settings → Secrets), TOML format:
   ```toml
   API_BASE_URL = "https://<your-render-service>.onrender.com"
   ```
   The app reads this via `os.getenv("API_BASE_URL", ...)` — Streamlit
   Cloud injects secrets as environment variables automatically, no code
   change needed.
3. Deploy. First load may be slow if Render's free instance was idle —
   see the cold-start note above.

## Post-deploy checklist

- [ ] `/health` returns your real candidate count, and it **survives a
      Render restart** (proves Supabase persistence is actually wired, not
      just configured) — this is the check from the persistence step
      earlier in this project; re-verify it against the deployed URL
      specifically, not just localhost.
- [ ] Run `python scripts/run_evaluation.py --api-base-url
      https://<your-render-service>.onrender.com` against your ground
      truth — confirms the deployed instance behaves the same as local.
- [ ] Confirm the Streamlit UI's near-miss "Check now" buttons and the
      candidate picker in "Check a specific candidate" work against the
      live backend, not just localhost — session-state behavior is worth
      double-checking in a fresh browser session.

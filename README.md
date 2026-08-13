# Smarter ATS — RAG Resume Screening

A resume-screening system built on retrieval-augmented generation, designed
around three failure modes that naive "chunk + embed + top-k" RAG doesn't
handle: **fragmentation**, **coverage**, and **hallucinated answers on
missing information**. See [`docs/architecture.md`](docs/architecture.md)
for the full design rationale.

Runs entirely on free tiers: local embedding + reranking models (no API
cost, no rate limit), Groq's free LLM tier for generation only, and
Supabase's free Postgres/pgvector tier for persistence.

## Status

This is a working skeleton: ingestion, indexing, retrieval, and evidence-gated
generation are implemented and tested end-to-end against synthetic data with
in-memory stores. Not yet wired: Supabase-backed persistence (currently
in-memory only — resets on restart) and deployment configs.

## Quickstart
python version 3.12 or 3.13 prefered
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY at minimum

# Generate a synthetic dataset (no real PII needed)
python scripts/generate_synthetic_resumes.py --count 100

# Run tests (no API key required — the unknowns-gate test proves this)
pytest

# Start the API
uvicorn src.api.main:app --reload --port 8000

# In another terminal, start the UI
streamlit run streamlit_app/app.py
```

Then in the UI: click **Ingest**, enter a query and a few screening
questions, and click **Run screening**.

## Project structure

```
src/
  ingestion/    parsing + chunking (tags every chunk with candidate_id)
  indexing/     Tier 1 (candidate-level) and Tier 2 (chunk-level) stores
  retrieval/    two-pass coverage pipeline + cross-encoder reranker
  generation/   evidence-gated LLM calls (the unknowns fix)
  api/          FastAPI app
streamlit_app/  demo UI showing evidence per candidate, not just verdicts
scripts/        synthetic resume generation
tests/          one test per core problem (fragmentation/coverage/unknowns)
docs/           architecture writeup
```

## Free-tier stack

| Layer | Tool | Cost |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (local) | Free, no rate limit |
| Reranker | `ms-marco-MiniLM-L-6-v2` (local) | Free, no rate limit |
| LLM | Groq (`llama-3.1-8b-instant`) | Free tier |
| Vector store | Supabase pgvector | Free tier |
| Backend hosting | Render free web service | Free |
| UI hosting | Streamlit Community Cloud | Free |

## Roadmap / TODO

- [ ] Implement `src/db/supabase_client.py` and Tier1/Tier2 store variants
      backed by Supabase, matching the `InMemoryTier1Store` /
      `InMemoryTier2Store` interfaces.
- [ ] Replace naive truncated summaries in `/ingest` with LLM-generated
      candidate summaries for better Tier 1 recall.
- [ ] Deploy backend to Render, UI to Streamlit Community Cloud.
- [ ] Add a bias/fairness note to docs — any resume-screening tool needs an
      explicit statement of what it does and doesn't account for.

## Why these three problems, specifically

See [`docs/architecture.md`](docs/architecture.md) — each fix is explained
with the specific failure mode it addresses and points to the exact module
implementing it.

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

Persistent (Supabase pgvector), hybrid dense+keyword retrieval, and
evidence-gated generation are implemented and tested against synthetic
data. Two retrieval paths exist: `/screen` (broad ranked search) and
`/screen-candidate` (direct check on a named candidate, bypassing ranking —
see "Coverage gap" in `docs/architecture.md` for why this exists). Near-miss
surfacing and a zero-cost score-calibration tool (`/score-preview` +
`scripts/analyze_confidence_scores.py`) round out the coverage/confidence
story. Not yet done: cloud deployment (see below) and the known limitation
noted just under this section.

### Known limitation — evidence-gate score distribution

The evidence gate's semantic signal (a cross-encoder score) cleanly
separates most real matches from no-evidence cases, but roughly the bottom
third of true-positive cases still score in a range that overlaps with
genuine no-evidence cases. `EVIDENCE_THRESHOLD` is currently set
conservatively (`0.15`, above the observed no-evidence ceiling) to favor
**safe UNKNOWNs over confident wrong answers** — the trade-off this project
is explicitly built around. The practical effect: some true positives in
that low-scoring tail will be answered UNKNOWN rather than YES/NO. This is
a known, accepted trade for now, not a resolved issue — see
`docs/architecture.md` for the calibration data behind this decision.
Planned follow-up (not yet started): inspect the low-scoring true-positive
cases individually to determine whether they're a data-labeling issue or a
genuine model limitation, and consider the embedding-model swap option
(asymmetric retrieval model) discussed in `docs/architecture.md` if the gap
persists.

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY, SUPABASE_DB_URL, USE_SUPABASE=true

# Generate a synthetic dataset (no real PII needed)
python scripts/generate_synthetic_resumes.py --count 100

# Run tests (no Groq key required — the unknowns-gate and evidence-scoring
# tests prove the gate's logic without spending any LLM calls)
pytest

# Start the API
uvicorn src.api.main:app --reload

# In another terminal, start the UI
streamlit run streamlit_app/app.py
```

Then in the UI: click **Ingest**, enter a query and a few screening
questions, and click **Run screening**. Use **"Check a specific
candidate"** to look someone up directly, bypassing ranking.

## Deployment

See [`docs/deployment.md`](docs/deployment.md) for step-by-step Render
(backend) and Streamlit Community Cloud (frontend) setup.

## Project structure

```
src/
  ingestion/    parsing + chunking (tags every chunk with candidate_id)
  indexing/     in-memory Tier 1/Tier 2 stores (local dev / USE_SUPABASE=false)
  db/           Supabase-backed Tier 1/Tier 2 stores + connection handling
  retrieval/    two-pass coverage pipeline, RRF hybrid search, near-miss logic, reranker
  generation/   evidence-gated LLM calls (the unknowns fix) + keyword matching
  api/          FastAPI app (/screen, /screen-candidate, /score-preview, /candidates, /ingest)
streamlit_app/  demo UI: ranked search, direct candidate check, near-miss surfacing
scripts/        synthetic resume generation, evaluation harness, score calibration
tests/          one test file per core problem/component
docs/           architecture writeup, deployment guide, SQL schema/migrations
```

## Free-tier stack

| Layer | Tool | Cost |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (local) | Free, no rate limit |
| Reranker / evidence scoring | `ms-marco-MiniLM-L-6-v2` (local, cross-encoder) | Free, no rate limit |
| LLM | Groq (`llama-3.1-8b-instant`) | Free tier |
| Vector store | Supabase pgvector | Free tier |
| Backend hosting | Render free web service | Free |
| UI hosting | Streamlit Community Cloud | Free |

## Roadmap / TODO

- [x] Supabase-backed Tier1/Tier2 stores (persistence)
- [x] Hybrid search (dense + keyword FTS via RRF)
- [x] Custom evaluation harness (hallucination-rate-first metrics)
- [x] Direct-candidate path (`/screen-candidate`) — closes the Coverage gap
      between exhaustive scoring and bounded generation
- [x] Near-miss surfacing in the UI
- [x] Cross-encoder evidence scoring (replaced cosine similarity — see
      "Known limitation" above for what's still open)
- [ ] Deploy backend to Render, UI to Streamlit Community Cloud
- [ ] Replace naive truncated summaries in `/ingest` with LLM-generated
      candidate summaries for better Tier 1 recall
- [ ] Investigate the low-scoring true-positive tail (see "Known limitation")
- [ ] Add a bias/fairness note — any resume-screening tool needs an
      explicit statement of what it does and doesn't account for

## Why these three problems, specifically

See [`docs/architecture.md`](docs/architecture.md) — each fix is explained
with the specific failure mode it addresses and points to the exact module
implementing it.
# Architecture

## Why naive RAG fails for resume screening

A standard RAG pipeline — chunk everything, embed, retrieve top-k globally,
stuff into an LLM prompt — breaks in three specific ways for this use case.

### 1. Fragmentation

Top-k chunk retrieval returns isolated sentences, not candidates. A resume
where "led a team of five engineers" appears once, buried among forty chunks
about unrelated technical work, may never surface that sentence in a flat
top-k search — even though the candidate is a strong match for a leadership
question.

**Fix:** two-tier index.
- Tier 1: one summary vector per candidate.
- Tier 2: chunks tagged with `candidate_id`.

Retrieval identifies *who* at Tier 1, then pulls *all* of that candidate's
chunks from Tier 2 — never an isolated fragment divorced from the rest of
their resume. See `src/indexing/tier1_index.py` and `tier2_index.py`.

### 2. Coverage

Ranking all chunks across all candidates and taking the top 4 systematically
under-represents candidates whose best-matching chunk isn't in the global
top 4, even if the rest of their resume is a strong fit. This is a
correctness bug, not a tuning problem — no top-k value fixes it, since the
count is being applied across the wrong axis (chunks, not candidates).

**Fix:** two-pass scoring.
1. Score every candidate (not chunk) with a cheap local embedding model.
   `Tier1Store.score_all()` is exhaustive — no cutoff.
2. Rerank only the shortlist (`SHORTLIST_SIZE`, default 25) with a
   cross-encoder for accurate relevance. See `src/retrieval/reranker.py`.

This keeps the pipeline free-tier-viable: the expensive step runs on a
bounded shortlist, not the full candidate pool.

### 3. Handling unknowns

An LLM asked "does this candidate have AWS experience?" will often infer a
plausible-sounding "yes" from adjacent context (e.g. "worked at a cloud
company") even when the resume never states it. This is the accuracy risk
that matters most for a hiring tool.

**Fix:** evidence-gated generation, enforced structurally, not just by
prompt instruction.
- Every question is scored against the candidate's chunks specifically for
  that question (not the original search query).
- If no chunk clears `EVIDENCE_THRESHOLD`, the LLM is never called — the
  result is hard-coded `UNKNOWN`.
- If it clears the threshold, only the qualifying excerpts are passed to the
  LLM, with a system prompt that requires citing evidence or returning
  UNKNOWN. See `src/generation/generator.py`.

## Persistence

Two interchangeable store implementations exist, both matching the same
interface (`upsert`/`score_all`/`get`/`all_ids` for Tier 1;
`add_chunks`/`chunks_for_candidate` for Tier 2):

- `InMemoryTier1Store` / `InMemoryTier2Store` — in-process, resets on
  restart. Used when `USE_SUPABASE=false` (default). Fine for local dev.
- `SupabaseTier1Store` / `SupabaseTier2Store` — persistent, backed by
  Postgres + pgvector on Supabase's free tier. Used when `USE_SUPABASE=true`.

### Why direct `psycopg2`, not `supabase-py`

`supabase-py`'s query builder is a REST wrapper and doesn't expose
`ORDER BY embedding <=> %s` (pgvector's cosine-distance operator) without
first writing a Postgres RPC function. Going direct via `psycopg2` +
`pgvector.psycopg2.register_vector` keeps the similarity query as plain,
readable, debuggable SQL instead of hiding it behind an RPC call.

### Why no ANN index

Tier 1's `score_all()` is deliberately exhaustive — that's the Coverage fix.
Approximate-nearest-neighbor indexes (ivfflat/hnsw) are built to skip most
rows for speed, which is exactly what Coverage is designed to prevent unless
you tune `probes` very aggressively. At hundreds of candidates, brute-force
cosine distance over every row is fast enough. Revisit only if the candidate
count grows into the thousands.

### Connection pooling

Use Supabase's **connection pooler** URI (port 6543, "Transaction" mode) for
`SUPABASE_DB_URL`, not the direct connection (port 5432). Free-tier direct
connections are capped low, and a Render-hosted API reconnecting frequently
can exhaust that cap. Find the pooler URI under Project Settings → Database
→ Connection pooling.

## Data flow

```
resumes (PDF/txt)
   │
   ▼
parser.py ── chunker.py (tags candidate_id + section_hint)
   │                         │
   ▼                         ▼
Tier 1 (summary+meta)   Tier 2 (chunks)
   │                         │
   └──── retrieve_candidates() ────┘
              │
   pass 1: score_all()      (Coverage — exhaustive)
   pass 2: rerank()         (Coverage — accurate, bounded)
   pass 3: chunks_for_candidate()  (Fragmentation — full context)
              │
              ▼
   answer_question_for_candidate()  (Unknowns — evidence gate)
              │
              ▼
        JSON verdict + evidence + confidence, per candidate, per question
```

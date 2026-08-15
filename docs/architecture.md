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

## Evaluation

`scripts/run_evaluation.py` scores the live system against a ground-truth
set you define — see `data/eval/ground_truth.json`. This is a custom
harness, not Ragas/TruLens: it grades against the project's own
deterministic evidence gate rather than using an LLM-as-judge, which avoids
both the free-tier Groq rate-limit exposure an LLM-judge framework would
add, and the awkwardness of grading an anti-hallucination system with
another LLM that can itself hallucinate a score.

The metric that matters most here isn't accuracy — it's the confusion
breakdown, specifically **hallucination rate** (expected UNKNOWN, system
answered anyway). That's the one failure mode the evidence gate exists to
prevent; a system that's very accurate on clear-cut cases but hallucinates
on ambiguous ones is worse for hiring than one with lower raw accuracy and
zero hallucinations.

```bash
# 1. Fill data/eval/ground_truth.json with real (query, candidate_id,
#    question, expected_verdict) cases — expected_verdict must come from
#    actually reading the candidate's resume, not guessing.
# 2. Make sure the API is running and candidates are ingested.
python scripts/run_evaluation.py
```

## Coverage gap between scoring and generation — and the direct-candidate fix

Tier 1's `score_all()` is exhaustive at the *scoring* stage, but `retrieve_candidates()`
only carries the top `shortlist_size` candidates into reranking and the
evidence gate. A candidate who scores low against a broad search query never
reaches generation at all — even for a question that has nothing to do with
that query. This surfaced concretely in evaluation as "not_retrieved" cases:
candidates silently missing from results, not answered incorrectly.

**Fix:** `/screen-candidate` (`get_direct_candidate_evidence()` in
`src/retrieval/pipeline.py`) bypasses Tier 1 shortlisting entirely for a
named candidate — no scoring, no reranking, straight to the evidence gate
against every chunk that candidate has. Use `/screen` for "who matches this
requirement" and `/screen-candidate` for "does this specific person meet
this requirement" — they're different tasks with different failure modes,
not the same task with an optional parameter.

**Closing the loop in the product, not just the backend:** a working
endpoint isn't enough — an HR user who doesn't see someone they expected in
`/screen` results has no way to know that person exists or that a direct
check is possible. `/screen` now returns a `coverage` block
(`total_candidates_indexed`, `shown_in_detail`, `not_evaluated_in_detail`),
and the UI surfaces it as a banner pointing at the direct-candidate lookup
when candidates were excluded. `/candidates` lists all ingested candidate
IDs so the UI can offer a picker instead of requiring the user to already
know a raw ID — a real deployment would want this to expose candidate
*names*, not database IDs, which this scaffold doesn't attempt.

`scripts/run_evaluation.py` now automatically retries any `/screen` miss
through `/screen-candidate` before counting it as `not_retrieved`, and
reports a `recovered_via_direct` count separately — this is what actually
proves the fix closes the loop end-to-end, rather than just existing as an
untested endpoint.

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
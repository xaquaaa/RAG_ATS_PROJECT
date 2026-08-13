"""
FastAPI entrypoint. Run with: uvicorn src.api.main:app --reload

NOTE: this wires the in-memory stores for local dev / demo purposes. Swap
InMemoryTier1Store / InMemoryTier2Store for the Supabase-backed versions in
src/db/ before deploying if you need persistence across restarts.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.ingestion.parser import load_all_resumes
from src.ingestion.chunker import chunk_resume
from src.retrieval.pipeline import retrieve_candidates
from src.generation.generator import answer_question_for_candidate

app = FastAPI(title="Smarter ATS — RAG Resume Screening")

# Store selection: USE_SUPABASE=true in .env switches to persistent storage.
# Both implementations share the same interface (upsert/score_all/get/all_ids
# for tier1; add_chunks/chunks_for_candidate for tier2) — nothing else here
# needs to change when you switch.
if settings.use_supabase:
    from src.db.tier1_supabase import SupabaseTier1Store
    from src.db.tier2_supabase import SupabaseTier2Store
    tier1 = SupabaseTier1Store()
    tier2 = SupabaseTier2Store()
else:
    from src.indexing.tier1_index import InMemoryTier1Store
    from src.indexing.tier2_index import InMemoryTier2Store
    tier1 = InMemoryTier1Store()
    tier2 = InMemoryTier2Store()


class ScreenRequest(BaseModel):
    query: str  # e.g. "5+ years backend experience with distributed systems"
    questions: list[str]  # explicit yes/no/unknown criteria, e.g. "Has AWS certification?"
    shortlist_size: int = 25


class IngestRequest(BaseModel):
    resume_directory: str = "data/resumes"


@app.post("/ingest")
def ingest(req: IngestRequest):
    """Loads resumes from disk, builds Tier 1 + Tier 2 indexes. Run once before querying."""
    resumes = load_all_resumes(req.resume_directory)
    if not resumes:
        raise HTTPException(400, f"No resumes found in {req.resume_directory}")

    for candidate_id, raw_text in resumes.items():
        # naive summary for the scaffold — replace with an LLM-generated
        # summary for better Tier 1 recall once you're past the skeleton.
        summary = raw_text[:1500]
        tier1.upsert(candidate_id, summary)
        tier2.add_chunks(chunk_resume(candidate_id, raw_text))

    return {"ingested_candidates": len(resumes)}


@app.post("/screen")
def screen(req: ScreenRequest):
    """
    Runs the full pipeline: Coverage-safe candidate retrieval, then
    evidence-gated question answering per candidate per question.
    """
    if not tier1.all_ids():
        raise HTTPException(400, "No candidates indexed yet — call /ingest first")

    candidates = retrieve_candidates(req.query, tier1, tier2, shortlist_size=req.shortlist_size)

    results = []
    for candidate in candidates:
        answers = {q: answer_question_for_candidate(q, candidate) for q in req.questions}
        results.append({
            "candidate_id": candidate.candidate_id,
            "tier1_score": round(candidate.tier1_score, 3),
            "rerank_score": round(candidate.rerank_score, 3) if candidate.rerank_score is not None else None,
            "answers": answers,
        })

    return {"query": req.query, "results": results}


@app.get("/health")
def health():
    return {"status": "ok", "candidates_indexed": len(tier1.all_ids())}

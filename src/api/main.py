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

    # Already sorted by rerank_score descending in retrieve_candidates().
    candidates = retrieve_candidates(req.query, tier1, tier2, shortlist_size=req.shortlist_size)
    total = len(candidates)

    results = []
    for rank, candidate in enumerate(candidates, start=1):
        answers = {q: answer_question_for_candidate(q, candidate) for q in req.questions}

        # User-facing signal: relative position in THIS shortlist, not a raw
        # logit. "Top 12%" is meaningful to a non-technical reviewer; a raw
        # cross-encoder score (which can be negative or >1) is not.
        match_percentile = round((rank / total) * 100) if total else None

        results.append({
            "candidate_id": candidate.candidate_id,
            "match_percentile": match_percentile,  # default-view field
            "rank": rank,
            "answers": {
                q: {
                    "verdict": a.get("verdict"),
                    "evidence": a.get("evidence"),
                    "confidence_label": _confidence_label(a.get("verdict"), a.get("top_score")),
                    "raw": a,  # full technical payload, shown only behind the UI's "show details" toggle
                }
                for q, a in answers.items()
            },
            "technical": {
                "tier1_score": round(candidate.tier1_score, 3),
                "rerank_score_raw": round(candidate.rerank_score, 3) if candidate.rerank_score is not None else None,
            },
        })

    return {"query": req.query, "results": results}


def _confidence_label(verdict: str | None, top_score: float | None) -> str | None:
    """
    Buckets retrieval confidence for display. Only meaningful for YES/NO
    verdicts — anything that reached the LLM already cleared
    EVIDENCE_THRESHOLD (the Unknowns gate), so this only differentiates
    strength ABOVE that floor. UNKNOWN verdicts show no confidence label —
    there's no evidence to rate.
    """
    if verdict == "UNKNOWN" or top_score is None:
        return None
    margin = top_score - settings.evidence_confidence_threshold
    if margin >= 0.20:
        return "High"
    elif margin >= 0.05:
        return "Medium"
    return "Low"


@app.get("/health")
def health():
    return {"status": "ok", "candidates_indexed": len(tier1.all_ids())}
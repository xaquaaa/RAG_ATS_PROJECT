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
from src.retrieval.pipeline import retrieve_candidates, get_direct_candidate_evidence
from src.generation.generator import answer_question_for_candidate, evaluate_evidence

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


class ScreenCandidateRequest(BaseModel):
    candidate_id: str
    questions: list[str]


class ScorePreviewRequest(BaseModel):
    candidate_id: str
    questions: list[str]


class SeedSyntheticDataRequest(BaseModel):
    count: int = 100


def _ingest_candidate(candidate_id: str, raw_text: str):
    """Shared by /ingest (reads from disk) and /admin/seed-synthetic-data
    (generates in-memory) — same indexing logic either way."""
    # naive summary for the scaffold — replace with an LLM-generated
    # summary for better Tier 1 recall once you're past the skeleton.
    summary = raw_text[:1500]
    tier1.upsert(candidate_id, summary)
    tier2.add_chunks(chunk_resume(candidate_id, raw_text))


@app.post("/ingest")
def ingest(req: IngestRequest):
    """Loads resumes from disk, builds Tier 1 + Tier 2 indexes. Run once before querying."""
    resumes = load_all_resumes(req.resume_directory)
    if not resumes:
        raise HTTPException(400, f"No resumes found in {req.resume_directory}")

    for candidate_id, raw_text in resumes.items():
        _ingest_candidate(candidate_id, raw_text)

    return {"ingested_candidates": len(resumes)}


@app.post("/admin/seed-synthetic-data")
def seed_synthetic_data(req: SeedSyntheticDataRequest):
    """
    Generates and ingests synthetic resumes ENTIRELY IN MEMORY — no files
    written to disk at any point. Exists because Render's free tier has no
    shell access (so you can't run generate_synthetic_resumes.py directly
    on the deployed instance) AND its free-tier disk is ephemeral anyway
    (any files written there vanish on the next redeploy/restart), so
    "write resumes to disk, then /ingest from disk" was never going to be a
    durable answer for a live deployment — only Supabase persistence
    survives restarts, so this writes straight into that.

    Gated by ADMIN_ENDPOINTS_ENABLED (default off). This is a demo
    convenience for synthetic data only — never enable it on a deployment
    holding real candidate data, since it's unauthenticated.
    """
    if not settings.admin_endpoints_enabled:
        raise HTTPException(
            403,
            "Admin endpoints are disabled. Set ADMIN_ENDPOINTS_ENABLED=true "
            "(demo/synthetic-data use only) to enable this.",
        )

    from scripts.generate_synthetic_resumes import build_resume

    for i in range(req.count):
        candidate_id = f"candidate_{i}"
        _ingest_candidate(candidate_id, build_resume(i))

    return {"seeded_candidates": req.count}


@app.post("/screen")
def screen(req: ScreenRequest):
    """
    Runs the full pipeline: Coverage-safe candidate retrieval, then
    evidence-gated question answering per candidate per question.
    """
    if not tier1.all_ids():
        raise HTTPException(400, "No candidates indexed yet — call /ingest first")

    # retrieve_candidates() now returns candidates + near-misses together —
    # near_misses are excluded candidates whose Tier-1 score was close to
    # the shortlist cutoff, not everyone who was excluded.
    retrieval = retrieve_candidates(req.query, tier1, tier2, shortlist_size=req.shortlist_size)
    candidates = retrieval.candidates
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

    total_indexed = len(tier1.all_ids())
    return {
        "query": req.query,
        "results": results,
        "coverage": {
            "total_candidates_indexed": total_indexed,
            "shown_in_detail": total,
            # Candidates that exist but were never scored past the shortlist
            # cutoff for THIS query — not wrong, just not evaluated here.
            "not_evaluated_in_detail": max(total_indexed - total, 0),
            # Near-misses only (not the full excluded list): candidates
            # whose Tier-1 score was within NEAR_MISS_MARGIN of the cutoff.
            # Capped at NEAR_MISS_MAX so a dense cluster near the boundary
            # doesn't flood the response — see docs/architecture.md.
            "near_misses": [
                {"candidate_id": nm.candidate_id, "margin_below_cutoff": nm.margin_below_cutoff}
                for nm in retrieval.near_misses
            ],
        },
    }


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
    if margin >= settings.confidence_high_margin:
        return "High"
    elif margin >= settings.confidence_medium_margin:
        return "Medium"
    return "Low"


@app.post("/screen-candidate")
def screen_candidate(req: ScreenCandidateRequest):
    """
    Direct-candidate path: checks a NAMED candidate against a set of
    questions, bypassing Tier 1 shortlisting entirely. Use this instead of
    /screen when the caller already knows who they want to check (e.g. "does
    this specific person meet these requirements") rather than searching
    broadly. /screen's shortlist_size cutoff means a candidate can be
    silently excluded from evaluation if they score low on a broad query —
    this endpoint never has that failure mode, since there's no scoring or
    shortlist step at all; every one of the candidate's chunks is available
    to the evidence gate for every question.

    Response shape intentionally mirrors /screen's per-candidate answer
    format (verdict/evidence/confidence_label/raw) so the UI can reuse the
    same rendering code — but there's no match_percentile or rank, since
    those only mean something relative to a ranked shortlist, which this
    path doesn't produce.
    """
    candidate = get_direct_candidate_evidence(req.candidate_id, tier1, tier2)
    if candidate is None:
        raise HTTPException(
            404,
            f"Candidate '{req.candidate_id}' not found — check the ID or call /ingest first",
        )

    answers = {q: answer_question_for_candidate(q, candidate) for q in req.questions}

    return {
        "candidate_id": req.candidate_id,
        "answers": {
            q: {
                "verdict": a.get("verdict"),
                "evidence": a.get("evidence"),
                "confidence_label": _confidence_label(a.get("verdict"), a.get("top_score")),
                "raw": a,
            }
            for q, a in answers.items()
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "candidates_indexed": len(tier1.all_ids())}


@app.get("/candidates")
def list_candidates():
    """
    Full candidate ID list, for populating a lookup picker in the UI so
    users aren't required to already know a raw candidate_id to use
    /screen-candidate. NOTE: this scaffold's synthetic dataset uses IDs like
    'candidate_46' as a stand-in for a name — a real deployment would want
    this endpoint to return actual candidate names, not just IDs, since no
    real HR user thinks in terms of database row identifiers.
    """
    return {"candidate_ids": sorted(tier1.all_ids())}


@app.post("/score-preview")
def score_preview(req: ScorePreviewRequest):
    """
    Runs the evidence gate (evaluate_evidence) WITHOUT calling the LLM —
    zero Groq token cost. Returns raw top_score and matched_via for each
    question against a named candidate, bypassing Tier 1 shortlisting the
    same way /screen-candidate does.

    Exists for calibrating EVIDENCE_THRESHOLD and the confidence-label
    margins (_confidence_label) against real score distributions instead of
    guessed constants — see scripts/analyze_confidence_scores.py, which
    sweeps this endpoint across many (candidate, question) pairs for free.
    """
    candidate = get_direct_candidate_evidence(req.candidate_id, tier1, tier2)
    if candidate is None:
        raise HTTPException(
            404,
            f"Candidate '{req.candidate_id}' not found — check the ID or call /ingest first",
        )

    scores = {}
    for q in req.questions:
        gate = evaluate_evidence(q, candidate)
        scores[q] = {
            "top_score": round(gate["top_score"], 4),
            "passes_gate": gate["passes"],
            "matched_via": gate["matched_via"],
        }

    return {"candidate_id": req.candidate_id, "scores": scores}

"""
End-to-end retrieval pipeline. This is where Fragmentation and Coverage fixes
compose. Handling Unknowns is applied downstream in generation.py, using the
per-candidate evidence this pipeline returns.
"""
from dataclasses import dataclass

from src.config import settings
from src.indexing.tier1_index import InMemoryTier1Store
from src.indexing.tier2_index import InMemoryTier2Store, ScoredChunk
from src.retrieval.reranker import rerank


@dataclass
class CandidateEvidence:
    candidate_id: str
    tier1_score: float
    rerank_score: float | None
    evidence_chunks: list[ScoredChunk]  # full-resume chunks for this candidate, ranked


def retrieve_candidates(
    query: str,
    tier1: InMemoryTier1Store,
    tier2: InMemoryTier2Store,
    shortlist_size: int = settings.shortlist_size,
) -> list[CandidateEvidence]:
    # Pass 1 — Coverage fix: score EVERY candidate, cheaply. No candidate is
    # skipped at this stage; this is what "coverage" actually means.
    all_scores = tier1.score_all(query)  # [(candidate_id, score), ...] for ALL candidates

    # Shortlist for the expensive step. Only pass-2 is bounded, not pass-1.
    shortlist = all_scores[:shortlist_size]

    # Pass 2 — rerank the shortlist with the cross-encoder for real relevance.
    summaries = [(cid, tier1.get(cid).summary_text) for cid, _ in shortlist if tier1.get(cid)]
    reranked = dict(rerank(query, summaries))

    results = []
    for cid, tier1_score in shortlist:
        # Fragmentation fix: pull the FULL set of this candidate's chunks,
        # not just whatever isolated chunk matched the query globally.
        evidence_chunks = tier2.chunks_for_candidate(cid, query, top_k=None)
        results.append(
            CandidateEvidence(
                candidate_id=cid,
                tier1_score=tier1_score,
                rerank_score=reranked.get(cid),
                evidence_chunks=evidence_chunks,
            )
        )

    results.sort(key=lambda r: (r.rerank_score or 0.0), reverse=True)
    return results


def get_direct_candidate_evidence(
    candidate_id: str,
    tier1: InMemoryTier1Store,
    tier2: InMemoryTier2Store,
) -> CandidateEvidence | None:
    """
    Direct-candidate path: bypasses Tier 1 shortlisting entirely for a named
    candidate. Exists because retrieve_candidates() is only exhaustive at
    the SCORING stage (Tier 1 scores every candidate) — the shortlist_size
    cutoff before reranking means a candidate who scores low against a
    broad search query never reaches the evidence gate at all, even for a
    question completely unrelated to that query. That's the exact failure
    the eval harness surfaced as "not_retrieved" — a candidate silently
    never considered, not a wrong answer.

    Use this when the caller already knows WHO they want to check (an HR
    user pulling up a specific person against a specific requirement) as
    opposed to WHO matches a broad search — those are genuinely two
    different tasks, not one task with an optional parameter.

    Returns None if the candidate isn't in Tier 1 (never ingested / bad ID).
    tier1_score is a 1.0 sentinel (not a real score — no scoring happened)
    and rerank_score is None (no reranking happened) so callers/UI can
    distinguish this path from a ranked result.
    """
    if tier1.get(candidate_id) is None:
        return None

    evidence_chunks = tier2.all_chunks_for_candidate(candidate_id)
    return CandidateEvidence(
        candidate_id=candidate_id,
        tier1_score=1.0,
        rerank_score=None,
        evidence_chunks=evidence_chunks,
    )
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

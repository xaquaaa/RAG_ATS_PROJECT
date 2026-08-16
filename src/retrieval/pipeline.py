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


@dataclass
class NearMiss:
    candidate_id: str
    tier1_score: float
    # How close to the shortlist cutoff this candidate was — not an
    # absolute quality signal, purely relative to who WAS included.
    margin_below_cutoff: float


@dataclass
class RetrievalResult:
    candidates: list[CandidateEvidence]  # the shortlist, evaluated in full
    near_misses: list[NearMiss]          # excluded, but close enough to flag
    total_candidates_scored: int         # size of the full candidate pool (Tier 1 pass-1 was exhaustive over this)


def compute_near_misses(
    all_scores: list[tuple[str, float]],
    shortlist_size: int,
    margin: float,
    max_count: int,
) -> list[NearMiss]:
    """
    Pure selection logic, split out from retrieve_candidates() so it's
    testable without a live embedding model. all_scores must already be
    sorted descending by score (as tier1.score_all() returns it).
    """
    shortlist = all_scores[:shortlist_size]
    excluded = all_scores[shortlist_size:]
    if not shortlist:
        return []

    cutoff_score = shortlist[-1][1]
    near_misses: list[NearMiss] = []
    for cid, score in excluded:
        gap = cutoff_score - score
        if gap <= margin:
            near_misses.append(NearMiss(candidate_id=cid, tier1_score=score, margin_below_cutoff=round(gap, 4)))
        else:
            # Sorted descending, so the gap only grows from here — safe to stop.
            break
    return near_misses[:max_count]


def retrieve_candidates(
    query: str,
    tier1: InMemoryTier1Store,
    tier2: InMemoryTier2Store,
    shortlist_size: int = settings.shortlist_size,
) -> RetrievalResult:
    # Pass 1 — Coverage fix: score EVERY candidate, cheaply. No candidate is
    # skipped at this stage; this is what "coverage" actually means.
    all_scores = tier1.score_all(query)  # [(candidate_id, score), ...] for ALL candidates

    shortlist = all_scores[:shortlist_size]
    near_misses = compute_near_misses(all_scores, shortlist_size, settings.near_miss_margin, settings.near_miss_max)

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
    return RetrievalResult(
        candidates=results,
        near_misses=near_misses,
        total_candidates_scored=len(all_scores),
    )


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

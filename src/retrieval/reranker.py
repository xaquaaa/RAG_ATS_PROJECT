"""
Cross-encoder reranker — Coverage fix, pass 2.

Pass 1 (Tier1Store.score_all) is exhaustive but cheap/approximate (bi-encoder
cosine similarity). Pass 2 takes only the shortlist from pass 1 and scores it
properly with a cross-encoder, which is far more accurate but too slow to run
against every candidate. Running it only on the shortlist is what keeps this
free-tier-viable.
"""
from functools import lru_cache
from sentence_transformers import CrossEncoder

from src.config import settings


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


def rerank(query: str, candidate_summaries: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """
    candidate_summaries: [(candidate_id, summary_text), ...]
    Returns [(candidate_id, rerank_score), ...] sorted descending.
    """
    if not candidate_summaries:
        return []
    model = get_reranker()
    pairs = [(query, summary) for _, summary in candidate_summaries]
    scores = model.predict(pairs)
    ranked = sorted(
        zip((cid for cid, _ in candidate_summaries), scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return [(cid, float(score)) for cid, score in ranked]

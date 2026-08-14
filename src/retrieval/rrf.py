"""
Reciprocal Rank Fusion — combines multiple ranked lists (e.g. vector-search
ranking + keyword-search ranking) into one fused score per item, without
needing the two lists' raw scores to be on comparable scales (cosine
similarity and ts_rank_cd are NOT directly comparable — RRF sidesteps that
by only using rank position, not raw score).

score(id) = sum over each ranked list containing id of: 1 / (k + rank)

k=60 is the standard default from the original RRF paper (Cormack et al.) —
it dampens the influence of rank position at the extremes; not something
that typically needs tuning per-dataset.
"""


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """
    ranked_lists: each is a list of ids in descending relevance order.
    An id that appears in zero lists never appears in the output — this
    matches SQL FULL OUTER JOIN / COALESCE(...,0) behavior in the Supabase
    version.
    """
    scores: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores

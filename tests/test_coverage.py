"""
Proves the Coverage fix: Tier 1's score_all() returns a score for EVERY
indexed candidate — no candidate is silently dropped by a top-k cutoff at
the scoring stage. Cutoffs only happen later, deliberately, for the
expensive reranking pass.
"""
from src.indexing.tier1_index import InMemoryTier1Store


def test_score_all_covers_every_candidate():
    store = InMemoryTier1Store()
    for i in range(50):
        store.upsert(f"candidate_{i}", f"Generic resume summary number {i}.")

    scores = store.score_all("looking for a backend engineer")

    assert len(scores) == 50
    assert {cid for cid, _ in scores} == {f"candidate_{i}" for i in range(50)}

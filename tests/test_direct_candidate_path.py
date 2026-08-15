"""
Proves the actual point of the direct-candidate path: a candidate who would
NEVER surface in a shortlist for a given broad query (because their resume
has nothing to do with that query) still gets evaluated correctly when
checked directly by ID. This is precisely the gap the eval harness's
"not_retrieved" cases exposed in retrieve_candidates()/shortlist_size.
"""
from src.ingestion.chunker import chunk_resume
from src.indexing.tier1_index import InMemoryTier1Store
from src.indexing.tier2_index import InMemoryTier2Store
from src.retrieval.pipeline import get_direct_candidate_evidence, retrieve_candidates


def _setup():
    tier1 = InMemoryTier1Store()
    tier2 = InMemoryTier2Store()

    # candidate_java: nothing but Java/Spring content — irrelevant to a
    # "Python data science" search query, so it should score poorly in
    # Tier 1 and very plausibly miss a small shortlist.
    java_resume = " ".join(["Senior Java developer, Spring Boot, enterprise systems."] * 40)
    tier1.upsert("candidate_java", java_resume[:500])
    tier2.add_chunks(chunk_resume("candidate_java", java_resume))

    # Fill Tier 1 with many strong matches for the broad query, so
    # candidate_java is pushed out of a small shortlist.
    for i in range(30):
        summary = f"Python data scientist, pandas, scikit-learn, candidate {i}."
        tier1.upsert(f"candidate_py_{i}", summary)
        tier2.add_chunks(chunk_resume(f"candidate_py_{i}", summary * 20))

    return tier1, tier2


def test_candidate_excluded_from_shortlist_but_reachable_directly():
    tier1, tier2 = _setup()

    # Confirm the gap actually exists first: a small shortlist for an
    # unrelated broad query should NOT include candidate_java.
    ranked = retrieve_candidates("Python data science pandas", tier1, tier2, shortlist_size=5)
    ranked_ids = {c.candidate_id for c in ranked}
    assert "candidate_java" not in ranked_ids, (
        "Test setup assumption failed — candidate_java unexpectedly made the shortlist"
    )

    # The direct path must still find them and return real evidence.
    direct = get_direct_candidate_evidence("candidate_java", tier1, tier2)
    assert direct is not None
    assert len(direct.evidence_chunks) > 0
    assert all(c.candidate_id == "candidate_java" for c in direct.evidence_chunks)


def test_unknown_candidate_id_returns_none():
    tier1, tier2 = _setup()
    assert get_direct_candidate_evidence("candidate_does_not_exist", tier1, tier2) is None
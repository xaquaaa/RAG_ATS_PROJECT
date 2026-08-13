"""
Proves the Fragmentation fix: once a candidate is identified, retrieval
returns ALL their chunks (filtered by candidate_id), not just whichever
single chunk happens to score highest globally.
"""
from src.ingestion.chunker import chunk_resume
from src.indexing.tier2_index import InMemoryTier2Store


def test_tier2_returns_full_candidate_context():
    tier2 = InMemoryTier2Store()

    # Expanded to 250+ words to ensure it breaks across multiple chunks 
    # even when CHUNK_SIZE_TOKENS is set to a large value like 220.
    resume_a = " ".join(
        ["Python developer with heavy FastAPI experience backend systems development engineering."] * 45 
        + ["Led a team of five engineers at a fintech startup tracking metrics and shipping architecture."]
    )
    resume_b = " ".join(
        ["Java developer with heavy Spring Boot experience enterprise microservices engineering scale."] * 45
    )

    tier2.add_chunks(chunk_resume("candidate_a", resume_a))
    tier2.add_chunks(chunk_resume("candidate_b", resume_b))

    # Query about team leadership — likely to match only ONE chunk globally
    # if you did flat top-k search, since most of candidate_a's resume is
    # about Python/FastAPI, not leadership.
    results = tier2.chunks_for_candidate("candidate_a", "team leadership experience", top_k=None)

    # The fix: we get back every chunk belonging to candidate_a, not just
    # the one leadership-related sentence.
    assert len(results) > 1
    assert all(r.candidate_id == "candidate_a" for r in results)

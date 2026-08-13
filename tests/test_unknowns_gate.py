"""
Proves the Handling Unknowns fix structurally: when no evidence clears the
confidence threshold, the LLM is never called, and the result is a hard
UNKNOWN — not a hallucinated guess.

This test does NOT require a Groq API key, because a correctly-gated
question should short-circuit before any API call is made.
"""
from src.generation.generator import answer_question_for_candidate
from src.retrieval.pipeline import CandidateEvidence
from src.indexing.tier2_index import ScoredChunk


def test_low_confidence_short_circuits_to_unknown():
    candidate = CandidateEvidence(
        candidate_id="candidate_x",
        tier1_score=0.5,
        rerank_score=0.5,
        evidence_chunks=[
            ScoredChunk(
                chunk_id="candidate_x::chunk_0",
                candidate_id="candidate_x",
                section_hint="skills",
                text="Proficient in Python, SQL, and data visualization.",
                score=0.5,
            )
        ],
    )

    # Asking about something entirely unrelated to the resume content —
    # should never clear the threshold, so this must return UNKNOWN without
    # touching the network / requiring GROQ_API_KEY.
    result = answer_question_for_candidate("Does the candidate hold a pilot's license?", candidate)

    assert result["verdict"] == "UNKNOWN"
    assert result["evidence"] is None

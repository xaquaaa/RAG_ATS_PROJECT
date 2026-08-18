"""
Handling Unknowns fix.

The gate happens BEFORE the LLM is ever called: if no chunk for this
candidate clears evidence requirements for this specific question, we
short-circuit to UNKNOWN and never spend a token. This is deliberate — it's
not just a prompt instruction (which can be ignored by the model), it's a
structural guarantee.

Hybrid evidence: a chunk counts as evidence if EITHER its cross-encoder
relevance score to the question clears EVIDENCE_THRESHOLD, OR it contains an
exact keyword match for a significant term in the question (catches "AWS",
"CKA", tool versions — exact strings a semantic model can under-rank). The
two signals are fused by rank position (RRF), not by combining raw scores,
since the two aren't on comparable scales.

NOTE on the semantic signal: this used to be raw cosine similarity between
independently-embedded question/chunk vectors (a bi-encoder comparison).
That was swapped for a cross-encoder score (ms-marco-MiniLM-L-6-v2, the same
model already used for Tier-1 reranking — no new model, no extra download)
after calibration data showed heavy overlap between expected-answerable and
expected-UNKNOWN score distributions using cosine similarity: the lowest
real-match score (0.207) was BELOW the highest no-evidence score (0.471),
meaning no threshold on that signal could separate the two cases well.
Cross-encoders score (question, chunk) jointly rather than comparing two
independently-computed vectors, and are generally far more discriminative
for this kind of relevance judgment — at the cost of being slower per
comparison (can't be precomputed/cached like an embedding), which is fine
here since it only runs on a single candidate's already-small chunk set
inside the evidence gate, never across an entire candidate pool.

IMPORTANT: switching the scoring function invalidates any previously-tuned
EVIDENCE_THRESHOLD / CONFIDENCE_HIGH_MARGIN / CONFIDENCE_MEDIUM_MARGIN —
cross-encoder output is on a different scale than cosine similarity (mapped
through a sigmoid here to land back in [0,1] for readability, but the
distribution's shape is not the same). Re-run
scripts/analyze_confidence_scores.py after this change and re-tune from its
output rather than trusting the old numbers.
"""
import json
import math
from groq import Groq

from src.config import settings
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.generation.keyword_match import extract_keywords, keyword_hit_count
from src.retrieval.pipeline import CandidateEvidence
from src.retrieval.reranker import get_reranker
from src.retrieval.rrf import reciprocal_rank_fusion


def _client() -> Groq:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set — see .env.example")
    return Groq(api_key=settings.groq_api_key)


def sigmoid(x: float) -> float:
    """
    Maps a cross-encoder's raw logit into (0, 1). ms-marco-MiniLM-L-6-v2
    isn't trained with an explicit probability head, so this is a
    readability transform, not a calibrated probability — it keeps the
    number in a familiar range for thresholds/UI display, nothing more.
    Guards against overflow on extreme logits (rare, but cross-encoder
    outputs are unbounded).
    """
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def evaluate_evidence(question: str, candidate: CandidateEvidence) -> dict:
    """
    Pure evidence-gate logic, split out from answer_question_for_candidate so
    it's testable without a Groq API key / network call (though the
    cross-encoder model itself still needs to be downloaded once). Returns
    everything needed to either short-circuit to UNKNOWN or proceed to
    generation.
    """
    keywords = extract_keywords(question)

    if not candidate.evidence_chunks:
        return {"passes": False, "top_score": 0.0, "matched_via": None, "relevant_excerpts": []}

    chunks = candidate.evidence_chunks
    reranker = get_reranker()
    # Batched in one predict() call — far cheaper than scoring chunks one at
    # a time, and this only ever runs against one candidate's chunk set
    # (small), never the whole candidate pool.
    raw_scores = reranker.predict([(question, c.text) for c in chunks])

    per_chunk = []
    for c, raw_score in zip(chunks, raw_scores):
        cross_score = sigmoid(float(raw_score))
        hits = keyword_hit_count(c.text, keywords)
        per_chunk.append({"chunk": c, "cross_score": cross_score, "keyword_hits": hits})

    vector_rank_ids = [d["chunk"].chunk_id for d in sorted(per_chunk, key=lambda d: d["cross_score"], reverse=True)]
    keyword_rank_ids = [
        d["chunk"].chunk_id
        for d in sorted((d for d in per_chunk if d["keyword_hits"] > 0), key=lambda d: d["keyword_hits"], reverse=True)
    ]
    fused_scores = reciprocal_rank_fusion([vector_rank_ids, keyword_rank_ids])

    by_id = {d["chunk"].chunk_id: d for d in per_chunk}
    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    top = by_id[ranked_ids[0]]
    top_score = top["cross_score"]
    top_has_keyword_hit = top["keyword_hits"] > 0

    passes_semantic = top_score >= settings.evidence_confidence_threshold
    passes = passes_semantic or top_has_keyword_hit
    matched_via = None
    if passes:
        matched_via = "both" if (passes_semantic and top_has_keyword_hit) else ("keyword" if top_has_keyword_hit else "semantic")

    relevant_ids = [
        cid for cid in ranked_ids
        if by_id[cid]["cross_score"] >= settings.evidence_confidence_threshold or by_id[cid]["keyword_hits"] > 0
    ][:5]
    relevant_excerpts = [by_id[cid]["chunk"].text for cid in relevant_ids]

    return {
        "passes": passes,
        "top_score": top_score,
        "matched_via": matched_via,
        "relevant_excerpts": relevant_excerpts,
    }


def answer_question_for_candidate(question: str, candidate: CandidateEvidence) -> dict:
    gate = evaluate_evidence(question, candidate)

    if not gate["passes"]:
        return {
            "candidate_id": candidate.candidate_id,
            "verdict": "UNKNOWN",
            "evidence": None,
            "confidence": "low",
            "reason": "no chunk cleared the evidence threshold or matched an exact keyword",
            "top_score": round(gate["top_score"], 3),
            "matched_via": None,
        }

    client = _client()
    completion = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, gate["relevant_excerpts"])},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(completion.choices[0].message.content)
    except (json.JSONDecodeError, IndexError):
        parsed = {"verdict": "UNKNOWN", "evidence": None, "confidence": "low"}

    parsed["candidate_id"] = candidate.candidate_id
    parsed["top_score"] = round(gate["top_score"], 3)
    parsed["matched_via"] = gate["matched_via"]
    return parsed

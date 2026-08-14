"""
Handling Unknowns fix.

The gate happens BEFORE the LLM is ever called: if no chunk for this
candidate clears evidence requirements for this specific question, we
short-circuit to UNKNOWN and never spend a token. This is deliberate — it's
not just a prompt instruction (which can be ignored by the model), it's a
structural guarantee.

Hybrid evidence: a chunk counts as evidence if EITHER its cosine similarity
to the question clears EVIDENCE_THRESHOLD, OR it contains an exact keyword
match for a significant term in the question (catches "AWS", "CKA", tool
versions — exact strings a semantic embedding can under-rank). The two
signals are fused by rank position (RRF), not by combining raw scores,
since cosine similarity and a keyword hit-count aren't on comparable scales.
"""
import json
from groq import Groq

from src.config import settings
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.generation.keyword_match import extract_keywords, keyword_hit_count
from src.indexing.embeddings import embed_text, cosine_sim
from src.retrieval.pipeline import CandidateEvidence
from src.retrieval.rrf import reciprocal_rank_fusion


def _client() -> Groq:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set — see .env.example")
    return Groq(api_key=settings.groq_api_key)


def evaluate_evidence(question: str, candidate: CandidateEvidence) -> dict:
    """
    Pure evidence-gate logic, split out from answer_question_for_candidate so
    it's testable without a Groq API key / network call. Returns everything
    needed to either short-circuit to UNKNOWN or proceed to generation.
    """
    q_vec = embed_text(question)
    keywords = extract_keywords(question)

    if not candidate.evidence_chunks:
        return {"passes": False, "top_score": 0.0, "matched_via": None, "relevant_excerpts": []}

    per_chunk = []
    for c in candidate.evidence_chunks:
        cos = cosine_sim(q_vec, embed_text(c.text))
        hits = keyword_hit_count(c.text, keywords)
        per_chunk.append({"chunk": c, "cosine": cos, "keyword_hits": hits})

    vector_rank_ids = [d["chunk"].chunk_id for d in sorted(per_chunk, key=lambda d: d["cosine"], reverse=True)]
    keyword_rank_ids = [
        d["chunk"].chunk_id
        for d in sorted((d for d in per_chunk if d["keyword_hits"] > 0), key=lambda d: d["keyword_hits"], reverse=True)
    ]
    fused_scores = reciprocal_rank_fusion([vector_rank_ids, keyword_rank_ids])

    by_id = {d["chunk"].chunk_id: d for d in per_chunk}
    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    top = by_id[ranked_ids[0]]
    top_score = top["cosine"]
    top_has_keyword_hit = top["keyword_hits"] > 0

    passes_semantic = top_score >= settings.evidence_confidence_threshold
    passes = passes_semantic or top_has_keyword_hit
    matched_via = None
    if passes:
        matched_via = "both" if (passes_semantic and top_has_keyword_hit) else ("keyword" if top_has_keyword_hit else "semantic")

    relevant_ids = [
        cid for cid in ranked_ids
        if by_id[cid]["cosine"] >= settings.evidence_confidence_threshold or by_id[cid]["keyword_hits"] > 0
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

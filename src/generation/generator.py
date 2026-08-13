"""
Handling Unknowns fix.

The gate happens BEFORE the LLM is ever called: if no chunk for this
candidate clears the confidence threshold for this specific question, we
short-circuit to UNKNOWN and never spend a token. This is deliberate — it's
not just a prompt instruction (which can be ignored by the model), it's a
structural guarantee.
"""
import json
from groq import Groq

from src.config import settings
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.indexing.embeddings import embed_text, cosine_sim
from src.retrieval.pipeline import CandidateEvidence


def _client() -> Groq:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set — see .env.example")
    return Groq(api_key=settings.groq_api_key)


def answer_question_for_candidate(question: str, candidate: CandidateEvidence) -> dict:
    q_vec = embed_text(question)

    # Re-score this candidate's chunks specifically against the QUESTION
    # (not the original search query — a candidate can be shortlisted for
    # one reason and asked about an unrelated criterion).
    scored = sorted(
        candidate.evidence_chunks,
        key=lambda c: cosine_sim(q_vec, embed_text(c.text)),
        reverse=True,
    )

    top_score = cosine_sim(q_vec, embed_text(scored[0].text)) if scored else 0.0

    # --- The gate ---
    if not scored or top_score < settings.evidence_confidence_threshold:
        return {
            "candidate_id": candidate.candidate_id,
            "verdict": "UNKNOWN",
            "evidence": None,
            "confidence": "low",
            "reason": "no chunk cleared the evidence confidence threshold",
            "top_score": round(top_score, 3),
        }

    # Only chunks above a lower floor go to the LLM as context — no point
    # feeding it clearly-irrelevant text even for a supported answer.
    relevant_excerpts = [c.text for c in scored if cosine_sim(q_vec, embed_text(c.text)) >= settings.evidence_confidence_threshold][:5]

    client = _client()
    completion = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, relevant_excerpts)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(completion.choices[0].message.content)
    except (json.JSONDecodeError, IndexError):
        parsed = {"verdict": "UNKNOWN", "evidence": None, "confidence": "low"}

    parsed["candidate_id"] = candidate.candidate_id
    parsed["top_score"] = round(top_score, 3)
    return parsed

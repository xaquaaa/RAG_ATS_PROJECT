"""
Lightweight keyword extraction for the hybrid (vector + keyword) evidence
gate in generator.py. Deliberately not using Postgres FTS here — this runs
against chunks already pulled into Python memory (candidate.evidence_chunks),
so a SQL round-trip per question would be wasteful. Simple substring/word
matching is sufficient for "does this exact technical term appear."
"""
import re

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "has", "have", "had",
    "does", "do", "did", "candidate", "this", "that", "with", "in", "of",
    "to", "for", "and", "or", "on", "at", "as", "by", "it", "their",
}


def extract_keywords(text: str) -> list[str]:
    """Pulls significant tokens (len >= 3, not a stopword) out of a question."""
    tokens = re.findall(r"[A-Za-z0-9+#.]{3,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def keyword_hit_count(chunk_text: str, keywords: list[str]) -> int:
    lowered = chunk_text.lower()
    return sum(1 for kw in keywords if kw in lowered)

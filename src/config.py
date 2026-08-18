"""
Central config. All values are overridable via environment variables (.env).
Nothing here should require a paid tier.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Embeddings — local model, no API cost, no rate limit.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Reranker — local cross-encoder, used only on the shortlist (Coverage fix, pass 2).
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # LLM — used ONLY for the final gated-generation step, never for scoring.
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # Vector store — Supabase Postgres + pgvector (free tier, persistent).
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")
    supabase_db_url: str = os.getenv("SUPABASE_DB_URL", "")  # postgres connection string
    use_supabase: bool = os.getenv("USE_SUPABASE", "false").lower() == "true"

    # Coverage: how many candidates pass 1 (cheap, exhaustive) shortlists for pass 2 (reranker).
    shortlist_size: int = int(os.getenv("SHORTLIST_SIZE", "25"))

    # Unknowns: cosine similarity floor below which we never call the LLM for that criterion.
    evidence_confidence_threshold: float = float(os.getenv("EVIDENCE_THRESHOLD", "0.45"))

    # Hybrid search: Reciprocal Rank Fusion constant. 60 is the standard
    # default from the original RRF paper — high enough that a single
    # method's rank-1 result doesn't completely dominate the fused score.
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    # Near-miss surfacing: a candidate just below the shortlist cutoff is
    # flagged if their Tier-1 score is within this margin of the lowest
    # shortlisted score. Capped at near_miss_max entries so a dense cluster
    # near the cutoff doesn't flood the response.
    near_miss_margin: float = float(os.getenv("NEAR_MISS_MARGIN", "0.05"))
    near_miss_max: int = int(os.getenv("NEAR_MISS_MAX", "10"))

    # Chunking
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "220"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "40"))

    # Confidence label margins (above EVIDENCE_THRESHOLD) for the UI's
    # High/Medium/Low display — see _confidence_label() in src/api/main.py.
    # These are GUESSES, not calibrated — run
    # scripts/analyze_confidence_scores.py against your real data and set
    # these from its suggested output rather than trusting the defaults.
    confidence_high_margin: float = float(os.getenv("CONFIDENCE_HIGH_MARGIN", "0.20"))
    confidence_medium_margin: float = float(os.getenv("CONFIDENCE_MEDIUM_MARGIN", "0.05"))


settings = Settings()

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

    # Coverage: how many candidates pass 1 (cheap, exhaustive) shortlists for pass 2 (reranker).
    shortlist_size: int = int(os.getenv("SHORTLIST_SIZE", "25"))

    # Unknowns: cosine similarity floor below which we never call the LLM for that criterion.
    evidence_confidence_threshold: float = float(os.getenv("EVIDENCE_THRESHOLD", "0.45"))

    # Chunking
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "220"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "40"))


settings = Settings()

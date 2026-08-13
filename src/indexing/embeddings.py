"""
Local embedding model. Runs on CPU, no API key, no rate limit — this is what
makes exhaustive per-candidate scoring (the Coverage fix) affordable.
"""
from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # Vectors are already normalized by embed_texts, so dot product == cosine sim.
    return float(np.dot(a, b))

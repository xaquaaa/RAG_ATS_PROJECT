"""
Local embedding model. Runs on CPU, no API key, no rate limit — this is what
makes exhaustive per-candidate scoring (the Coverage fix) affordable.

Uses fastembed (ONNX runtime), not sentence-transformers (PyTorch), after
Render's free 512MB web service ran out of memory with two PyTorch-backed
models (this embedder + the cross-encoder in reranker.py/generator.py)
loaded simultaneously — PyTorch's baseline runtime overhead alone is
substantial before either model's own weights are counted. fastembed has no
PyTorch dependency at all.

Model choice: 'sentence-transformers/all-MiniLM-L6-v2' is directly supported
by fastembed's registry — it's an ONNX export of the SAME weights previously
used via sentence-transformers, not a different model. This means existing
cosine-similarity behavior (Tier 1 scoring, thresholds) should carry over
without needing recalibration, unlike the earlier cosine→cross-encoder
change which genuinely changed the scoring semantics. Still worth a quick
sanity check after deploying, not assumed identical to the last decimal.
"""
from functools import lru_cache
import numpy as np
from fastembed import TextEmbedding

from src.config import settings


@lru_cache(maxsize=1)
def get_model() -> TextEmbedding:
    return TextEmbedding(model_name=settings.embedding_model)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    # fastembed's embed() returns a generator of arrays, not guaranteed
    # pre-normalized (unlike sentence-transformers' normalize_embeddings=True
    # flag) — normalize explicitly so cosine_sim's dot-product shortcut
    # below stays valid.
    return np.array([_normalize(v) for v in model.embed(texts)])


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # Vectors are already normalized by embed_texts, so dot product == cosine sim.
    return float(np.dot(a, b))

"""
Tier 2 (chunk-level) index.

Stores every chunk with its candidate_id and embedding. The critical query
pattern here is NOT "top-k across all chunks" — it's "give me every chunk
for candidate X, ranked by relevance to this specific criterion." That
distinction is the whole Fragmentation fix: once Tier 1 has told you WHO to
look at, Tier 2 gives you their FULL resume, not just whichever isolated
sentence happened to score highest globally.

Supabase/pgvector schema:

    create table resume_chunks (
        chunk_id text primary key,
        candidate_id text references candidates(candidate_id),
        section_hint text,
        text text,
        embedding vector(384)
    );
    create index on resume_chunks (candidate_id);
"""
from dataclasses import dataclass
import numpy as np

from src.indexing.embeddings import embed_texts, embed_text, cosine_sim
from src.ingestion.chunker import Chunk


@dataclass
class ScoredChunk:
    chunk_id: str
    candidate_id: str
    section_hint: str
    text: str
    score: float


class InMemoryTier2Store:
    def __init__(self):
        self._chunks: list[dict] = []

    def add_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return
        vectors = embed_texts([c.text for c in chunks])
        for chunk, vec in zip(chunks, vectors):
            self._chunks.append({
                "chunk_id": chunk.chunk_id,
                "candidate_id": chunk.candidate_id,
                "section_hint": chunk.section_hint,
                "text": chunk.text,
                "embedding": vec,
            })

    def chunks_for_candidate(self, candidate_id: str, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        """
        This is the anti-fragmentation call: filter by candidate_id FIRST,
        then rank within that candidate's own chunks. top_k=None returns
        every chunk for the candidate — use that when you need the full
        picture (e.g. for the evidence-gating step in generation.py).
        """
        q_vec = embed_text(query)
        candidate_chunks = [c for c in self._chunks if c["candidate_id"] == candidate_id]
        scored = [
            ScoredChunk(
                chunk_id=c["chunk_id"],
                candidate_id=c["candidate_id"],
                section_hint=c["section_hint"],
                text=c["text"],
                score=cosine_sim(q_vec, c["embedding"]),
            )
            for c in candidate_chunks
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k] if top_k else scored

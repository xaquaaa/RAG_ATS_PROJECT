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

from src.config import settings
from src.indexing.embeddings import embed_texts, embed_text, cosine_sim
from src.ingestion.chunker import Chunk
from src.retrieval.rrf import reciprocal_rank_fusion


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
        Anti-fragmentation call: filter by candidate_id FIRST, then rank
        within that candidate's own chunks. Ranking is hybrid (vector +
        substring keyword match) via RRF, mirroring SupabaseTier2Store's SQL
        version, so behavior is consistent whether USE_SUPABASE is on or off.
        top_k=None returns every chunk for the candidate — use that when you
        need the full picture (as generator.py does for the evidence gate).
        """
        q_vec = embed_text(query)
        query_terms = [t for t in query.lower().split() if len(t) >= 3]
        candidate_chunks = [c for c in self._chunks if c["candidate_id"] == candidate_id]

        cosine_by_id = {c["chunk_id"]: cosine_sim(q_vec, c["embedding"]) for c in candidate_chunks}
        keyword_hits_by_id = {
            c["chunk_id"]: sum(1 for t in query_terms if t in c["text"].lower()) for c in candidate_chunks
        }

        vector_rank_ids = sorted(cosine_by_id, key=lambda cid: cosine_by_id[cid], reverse=True)
        keyword_rank_ids = sorted(
            (cid for cid, hits in keyword_hits_by_id.items() if hits > 0),
            key=lambda cid: keyword_hits_by_id[cid],
            reverse=True,
        )
        fused = reciprocal_rank_fusion([vector_rank_ids, keyword_rank_ids], k=settings.rrf_k)

        by_id = {c["chunk_id"]: c for c in candidate_chunks}
        ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)

        scored = [
            ScoredChunk(
                chunk_id=cid,
                candidate_id=by_id[cid]["candidate_id"],
                section_hint=by_id[cid]["section_hint"],
                text=by_id[cid]["text"],
                score=fused[cid],
            )
            for cid in ranked_ids
        ]
        return scored[:top_k] if top_k else scored

    def all_chunks_for_candidate(self, candidate_id: str) -> list[ScoredChunk]:
        """
        Used by the direct-candidate path (bypasses Tier 1 shortlisting
        entirely — see src/retrieval/pipeline.py get_direct_candidate_evidence).
        No query, no ranking: just every chunk this candidate has. Ranking
        against each specific question happens downstream in
        generator.answer_question_for_candidate, which re-scores per
        question anyway — pre-ranking here would be wasted work and would
        require a fake query, which is exactly the coupling this path exists
        to avoid.
        """
        return [
            ScoredChunk(
                chunk_id=c["chunk_id"],
                candidate_id=c["candidate_id"],
                section_hint=c["section_hint"],
                text=c["text"],
                score=0.0,
            )
            for c in self._chunks
            if c["candidate_id"] == candidate_id
        ]
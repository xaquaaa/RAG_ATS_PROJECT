"""
Supabase/pgvector-backed Tier 2 store. Same interface as InMemoryTier2Store
(add_chunks, chunks_for_candidate).
"""
from dataclasses import dataclass

from src.db.connection import get_connection
from src.indexing.embeddings import embed_texts, embed_text
from src.ingestion.chunker import Chunk


@dataclass
class ScoredChunk:
    chunk_id: str
    candidate_id: str
    section_hint: str
    text: str
    score: float


class SupabaseTier2Store:
    def add_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return
        vectors = embed_texts([c.text for c in chunks])
        with get_connection() as conn:
            with conn.cursor() as cur:
                for chunk, vec in zip(chunks, vectors):
                    cur.execute(
                        """
                        insert into resume_chunks (chunk_id, candidate_id, section_hint, text, embedding)
                        values (%s, %s, %s, %s, %s)
                        on conflict (chunk_id)
                        do update set text = excluded.text, embedding = excluded.embedding
                        """,
                        (chunk.chunk_id, chunk.candidate_id, chunk.section_hint, chunk.text, vec),
                    )

    def chunks_for_candidate(self, candidate_id: str, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        """
        Fragmentation fix: filter by candidate_id FIRST (indexed, cheap),
        then rank within that candidate's own chunks by similarity to the
        query/question. No global top-k across candidates happens here.
        """
        q_vec = embed_text(query)
        sql = (
            "select chunk_id, candidate_id, section_hint, text, 1 - (embedding <=> %s) as score "
            "from resume_chunks where candidate_id = %s order by score desc"
        )
        params = [q_vec, candidate_id]
        if top_k:
            sql += " limit %s"
            params.append(top_k)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return [
                    ScoredChunk(chunk_id=r[0], candidate_id=r[1], section_hint=r[2], text=r[3], score=float(r[4]))
                    for r in cur.fetchall()
                ]

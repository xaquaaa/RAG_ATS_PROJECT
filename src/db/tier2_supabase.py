"""
Supabase/pgvector-backed Tier 2 store. Same interface as InMemoryTier2Store
(add_chunks, chunks_for_candidate).
"""
from dataclasses import dataclass

from src.config import settings
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
        Fragmentation fix + Hybrid search: filter by candidate_id FIRST
        (indexed, cheap), then rank within that candidate's own chunks using
        Reciprocal Rank Fusion of two independent rankings:
          - vector_rank: cosine distance against the embedding
          - keyword_rank: Postgres full-text search (ts_rank_cd), which
            catches exact technical terms (AWS, CKA, tool versions) that a
            semantic embedding can under-rank if the surrounding phrasing
            is unusual.

        RRF combines them by RANK POSITION, not raw score — cosine
        similarity and ts_rank_cd are on incompatible scales, so summing
        them directly would silently let whichever metric has the larger
        numeric range dominate. This is done in one query so we're not
        round-tripping to Python for the fusion step.

        Uses the 'simple' text-search config (no stemming/stopword removal)
        rather than 'english' — stemming distorts exact technical tokens
        like acronyms and version strings, which is exactly what keyword
        search is here to catch precisely.
        """
        q_vec = embed_text(query)
        k = settings.rrf_k
        sql = """
            with vector_rank as (
                select chunk_id, rank() over (order by embedding <=> %(qvec)s) as vrank
                from resume_chunks
                where candidate_id = %(cid)s
            ),
            keyword_rank as (
                select chunk_id,
                       rank() over (order by ts_rank_cd(search_vector, plainto_tsquery('simple', %(qtext)s)) desc) as krank
                from resume_chunks
                where candidate_id = %(cid)s
                  and search_vector @@ plainto_tsquery('simple', %(qtext)s)
            )
            select rc.chunk_id, rc.candidate_id, rc.section_hint, rc.text,
                   coalesce(1.0 / (%(k)s + v.vrank), 0)
                   + coalesce(1.0 / (%(k)s + kw.krank), 0) as score
            from resume_chunks rc
            left join vector_rank v on rc.chunk_id = v.chunk_id
            left join keyword_rank kw on rc.chunk_id = kw.chunk_id
            where rc.candidate_id = %(cid)s
            order by score desc
        """
        params = {"qvec": q_vec, "cid": candidate_id, "qtext": query, "k": k}
        if top_k:
            sql += " limit %(top_k)s"
            params["top_k"] = top_k

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return [
                    ScoredChunk(chunk_id=r[0], candidate_id=r[1], section_hint=r[2], text=r[3], score=float(r[4]))
                    for r in cur.fetchall()
                ]

    def all_chunks_for_candidate(self, candidate_id: str) -> list[ScoredChunk]:
        """
        Used by the direct-candidate path — bypasses Tier 1 shortlisting and
        all ranking. See src/retrieval/pipeline.py get_direct_candidate_evidence
        and InMemoryTier2Store.all_chunks_for_candidate for the rationale.
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "select chunk_id, candidate_id, section_hint, text "
                    "from resume_chunks where candidate_id = %s",
                    (candidate_id,),
                )
                return [
                    ScoredChunk(chunk_id=r[0], candidate_id=r[1], section_hint=r[2], text=r[3], score=0.0)
                    for r in cur.fetchall()
                ]
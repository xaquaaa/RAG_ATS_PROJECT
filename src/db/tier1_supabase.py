"""
Supabase/pgvector-backed Tier 1 store. Same interface as InMemoryTier1Store
(upsert, score_all, get, all_ids) so src/api/main.py can swap between them
via one config flag — nothing else in the pipeline needs to change.
"""
import json
from dataclasses import dataclass
import numpy as np

from src.db.connection import get_connection
from src.indexing.embeddings import embed_text


@dataclass
class CandidateRecord:
    candidate_id: str
    summary_text: str
    embedding: np.ndarray
    metadata: dict


class SupabaseTier1Store:
    def upsert(self, candidate_id: str, summary_text: str, metadata: dict | None = None):
        embedding = embed_text(summary_text)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into candidates (candidate_id, summary_text, embedding, metadata)
                    values (%s, %s, %s, %s)
                    on conflict (candidate_id)
                    do update set summary_text = excluded.summary_text,
                                  embedding = excluded.embedding,
                                  metadata = excluded.metadata
                    """,
                    (candidate_id, summary_text, embedding, json.dumps(metadata or {})),
                )

    def score_all(self, query: str) -> list[tuple[str, float]]:
        """
        Coverage fix, pass 1 — exhaustive scan of every candidate row. No
        LIMIT, no ANN index: every candidate gets a real score against the
        query. See docs/architecture.md for why this is deliberate.

        `1 - (embedding <=> %s)` converts pgvector's cosine DISTANCE
        (0 = identical) into cosine SIMILARITY (1 = identical), matching the
        convention used by InMemoryTier1Store / cosine_sim().
        """
        q_vec = embed_text(query)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "select candidate_id, 1 - (embedding <=> %s) as score "
                    "from candidates order by score desc",
                    (q_vec,),
                )
                return [(row[0], float(row[1])) for row in cur.fetchall()]

    def get(self, candidate_id: str) -> CandidateRecord | None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "select candidate_id, summary_text, embedding, metadata "
                    "from candidates where candidate_id = %s",
                    (candidate_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return CandidateRecord(
                    candidate_id=row[0],
                    summary_text=row[1],
                    embedding=np.array(row[2]),
                    metadata=row[3] or {},
                )

    def all_ids(self) -> list[str]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("select candidate_id from candidates")
                return [row[0] for row in cur.fetchall()]

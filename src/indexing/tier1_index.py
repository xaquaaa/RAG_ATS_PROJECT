"""
Tier 1 (candidate-level) index.

One row per candidate: a summary embedding + structured metadata. This is the
index the Coverage pass-1 scan runs against — it's small (n_candidates rows,
not n_chunks rows) so scanning ALL of it per query is cheap.

Schema mirrors what you'd create in Supabase/pgvector:

    create table candidates (
        candidate_id text primary key,
        summary_text text,
        embedding vector(384),   -- 384 = all-MiniLM-L6-v2 output dim
        years_experience int,
        skills text[],
        education text
    );

This module ships an in-memory fallback (dict + numpy) so the pipeline is
runnable and testable before Supabase is wired up. Swap `InMemoryTier1Store`
for `SupabaseTier1Store` in src/db/ once you have credentials — same interface.
"""
from dataclasses import dataclass
import numpy as np

from src.indexing.embeddings import embed_text, cosine_sim


@dataclass
class CandidateRecord:
    candidate_id: str
    summary_text: str
    embedding: np.ndarray
    metadata: dict


class InMemoryTier1Store:
    def __init__(self):
        self._records: dict[str, CandidateRecord] = {}

    def upsert(self, candidate_id: str, summary_text: str, metadata: dict | None = None):
        self._records[candidate_id] = CandidateRecord(
            candidate_id=candidate_id,
            summary_text=summary_text,
            embedding=embed_text(summary_text),
            metadata=metadata or {},
        )

    def score_all(self, query: str) -> list[tuple[str, float]]:
        """
        Coverage fix, pass 1: score EVERY candidate against the query.
        Returns [(candidate_id, score), ...] sorted descending. Exhaustive by
        design — no top-k cutoff happens here.
        """
        q_vec = embed_text(query)
        return sorted(
            ((cid, cosine_sim(q_vec, rec.embedding)) for cid, rec in self._records.items()),
            key=lambda x: x[1],
            reverse=True,
        )

    def get(self, candidate_id: str) -> CandidateRecord | None:
        return self._records.get(candidate_id)

    def all_ids(self) -> list[str]:
        return list(self._records.keys())

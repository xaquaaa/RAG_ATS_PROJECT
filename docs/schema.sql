-- Run this in the Supabase SQL editor (Project → SQL Editor → New query).

create extension if not exists vector;

create table if not exists candidates (
    candidate_id text primary key,
    summary_text text not null,
    embedding vector(384) not null,   -- 384 = all-MiniLM-L6-v2 output dim
    metadata jsonb default '{}'::jsonb,
    created_at timestamptz default now()
);

create table if not exists resume_chunks (
    chunk_id text primary key,
    candidate_id text not null references candidates(candidate_id) on delete cascade,
    section_hint text,
    text text not null,
    embedding vector(384) not null,
    created_at timestamptz default now()
);

-- Indexed for the Fragmentation fix: fast "all chunks for this candidate" lookups.
-- Deliberately NOT adding a vector ANN index here or on candidates.embedding —
-- see docs/architecture.md "Why no ANN index" for the reasoning.
create index if not exists resume_chunks_candidate_id_idx on resume_chunks (candidate_id);

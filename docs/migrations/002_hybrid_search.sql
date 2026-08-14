-- Run this in the Supabase SQL editor AFTER docs/schema.sql has already run.
-- Adds keyword full-text search alongside the existing vector search, so
-- exact technical terms (AWS, CKA, "k8s") aren't at the mercy of semantic
-- embedding similarity alone.

alter table resume_chunks
    add column if not exists text_tsv tsvector
    generated always as (to_tsvector('english', text)) stored;

create index if not exists resume_chunks_text_tsv_idx
    on resume_chunks using gin (text_tsv);

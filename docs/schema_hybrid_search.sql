-- Migration: adds keyword full-text search to resume_chunks for hybrid
-- (vector + FTS) retrieval. Run this in the Supabase SQL editor AFTER
-- schema.sql has already been applied.

-- 'simple' config, not 'english': no stemming, no stopword removal. This
-- matters for exact technical-token matching (AWS, CKA, tool version
-- strings) — 'english' stemming can distort short acronyms and would strip
-- tokens that don't behave like English words.
alter table resume_chunks
    add column if not exists search_vector tsvector
    generated always as (to_tsvector('simple', text)) stored;

create index if not exists resume_chunks_search_vector_idx
    on resume_chunks using gin (search_vector);

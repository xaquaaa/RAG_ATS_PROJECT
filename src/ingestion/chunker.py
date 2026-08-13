"""
Chunks a resume into overlapping windows, tagging every chunk with candidate_id.

This tagging is the load-bearing part of the Fragmentation fix (see README /
docs/architecture.md). Without candidate_id on every chunk, Tier 2 retrieval
can't reliably pull "the rest of this candidate's resume" once Tier 1 has
shortlisted them.
"""
from dataclasses import dataclass, field
import re

from src.config import settings


@dataclass
class Chunk:
    candidate_id: str
    chunk_id: str
    text: str
    section_hint: str = "unknown"  # e.g. "experience", "education", "skills" — best-effort


SECTION_HEADERS = {
    "experience": r"(work experience|professional experience|experience)",
    "education": r"(education|academic background)",
    "skills": r"(skills|technical skills|technologies)",
    "projects": r"(projects|personal projects)",
    "summary": r"(summary|objective|profile)",
}


def _guess_section(text_block: str) -> str:
    lowered = text_block.lower()
    for section, pattern in SECTION_HEADERS.items():
        if re.search(pattern, lowered):
            return section
    return "unknown"


def _tokenize_words(text: str) -> list[str]:
    # Word-level windowing is good enough here; swap for a real tokenizer
    # (tiktoken) if you need exact token-count guarantees.
    return text.split()


def chunk_resume(candidate_id: str, raw_text: str) -> list[Chunk]:
    words = _tokenize_words(raw_text)
    size = settings.chunk_size_tokens
    overlap = settings.chunk_overlap_tokens
    step = max(size - overlap, 1)

    chunks: list[Chunk] = []
    idx = 0
    for start in range(0, len(words), step):
        window = words[start:start + size]
        if not window:
            continue
        text_block = " ".join(window)
        chunks.append(
            Chunk(
                candidate_id=candidate_id,
                chunk_id=f"{candidate_id}::chunk_{idx}",
                text=text_block,
                section_hint=_guess_section(text_block),
            )
        )
        idx += 1
        if start + size >= len(words):
            break
    return chunks

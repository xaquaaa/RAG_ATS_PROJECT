"""
Parses raw resume files (PDF or .txt) into plain text.
Keep this dumb on purpose — layout-aware parsing is a separate, harder problem
that isn't one of the three problems this project is about. Don't scope-creep here.
"""
from pathlib import Path
from pypdf import PdfReader


def parse_resume(file_path: str) -> str:
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif path.suffix.lower() in (".txt", ".md"):
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported resume format: {path.suffix}")


def load_all_resumes(directory: str) -> dict[str, str]:
    """Returns {candidate_id: raw_text}. candidate_id = filename stem."""
    out = {}
    for f in Path(directory).glob("*"):
        if f.suffix.lower() in (".pdf", ".txt", ".md"):
            out[f.stem] = parse_resume(str(f))
    return out

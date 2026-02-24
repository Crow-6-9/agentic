import re
import logging
from docx import Document

logger = logging.getLogger(__name__)


def extract_text_from_docx(file_path: str) -> str:
    """Extract clean text from a .docx file."""
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


def extract_metadata(text: str) -> dict:
    """
    Regex-based metadata extraction from resume text.

    Email → reliable regex match across the full text.
    Name  → first non-empty line heuristic.
    """
    email_match = re.search(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        text,
    )
    email = email_match.group(0) if email_match else "N/A"

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    raw   = lines[0] if lines else "Unknown"
    name  = raw if re.match(r"^[A-Za-z .'\-]{2,50}$", raw) else "Unknown"

    logger.debug(f"Metadata — name='{name}'  email='{email}'")
    return {"name": name, "email": email}


def build_searchable_chunk(text: str, metadata: dict) -> str:
    """
    Build a clean, consistent representation of a resume for embedding.

    WHY THIS APPROACH:
      Chainlit worked because both resume and query embeddings were in the
      same vector space (full text → full text). The fix isn't exotic chunking
      — it's keeping that consistency while removing noise that dilutes signals.

      Regex-based skill extraction breaks when skills appear in paragraph form:
        "I worked extensively with Hadoop clusters..."  → regex MISSES this
      So we don't filter lines. We clean the full text and trim it to a size
      where the embedding model captures the whole document cleanly.

    Strategy:
      1. Prepend identity + a skills-section hint so the embedding is anchored
      2. Clean the raw text (remove excessive whitespace, repeated chars)
      3. Trim to ~1200 words — enough for full context, short enough that
         no single skill gets buried below the model's attention window
      4. Result: one clean chunk per resume, same vector space as the query
    """
    name  = metadata.get("name", "Unknown")
    email = metadata.get("email", "N/A")

    # Clean the text — collapse blank lines, strip unicode noise
    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip lines that are just repeated characters (e.g. "------", "======")
        if re.match(r'^[\-=_\*\.]{3,}$', line):
            continue
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    # Trim to ~1200 words so the embedding isn't overloaded
    words = cleaned_text.split()
    if len(words) > 1200:
        cleaned_text = " ".join(words[:1200])

    # Prepend identity so metadata is part of the vector
    # This ensures name/email searches also work
    chunk = (
        f"Candidate: {name}\n"
        f"Email: {email}\n\n"
        f"{cleaned_text}"
    )

    logger.debug(f"Searchable chunk — {len(chunk.split())} words for '{name}'")
    return chunk


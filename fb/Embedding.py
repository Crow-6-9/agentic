import os
import re
import logging
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

API_KEY         = os.getenv("API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_URL   = os.getenv("EMBEDDING_URL", "https://api.openai.com/v1/embeddings")

# Most embedding models support 8192 tokens max.
# ~4 chars per token on average → 6000 words is safe for any provider.
MAX_WORDS = 6000


def _clean_text(text: str) -> str:
    """
    Clean extracted text before sending to the embedding API.

    PDF extraction produces noise that causes 400/403 errors:
      - Non-UTF8 / control characters  → API rejects the payload
      - Repeated whitespace/newlines   → inflates token count
      - Very long text                 → exceeds token limit
    """
    if not text:
        return ""

    # Remove non-printable / control characters (except newline and tab)
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", " ", text)

    # Collapse repeated whitespace and blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Truncate to MAX_WORDS — keeps well within any provider's token limit
    words = text.split()
    if len(words) > MAX_WORDS:
        logger.warning(f"Text truncated from {len(words)} to {MAX_WORDS} words before embedding")
        text = " ".join(words[:MAX_WORDS])

    return text


def get_embedding(text: str) -> list | None:
    """Call the configured embedding API. Returns None on failure."""
    if not API_KEY or not EMBEDDING_MODEL:
        logger.error("API_KEY or EMBEDDING_MODEL not set in .env")
        return None

    cleaned = _clean_text(text)

    if not cleaned:
        logger.error("Empty text after cleaning — skipping embedding")
        return None

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {"input": cleaned, "model": EMBEDDING_MODEL}

    try:
        resp = requests.post(EMBEDDING_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        vec = resp.json()["data"][0]["embedding"]
        logger.debug(f"Embedding OK — words={len(cleaned.split())} dim={len(vec)}")
        return vec
    except requests.HTTPError as e:
        logger.error(
            f"Embedding HTTP {e.response.status_code} — "
            f"words={len(cleaned.split())} "
            f"body={e.response.text[:300]}"
        )
        return None
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None

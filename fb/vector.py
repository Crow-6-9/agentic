"""
vectorstore.py — ChromaDB persistence layer.

Strategy : ONE resume = ONE document (one chunk).
Doc ID   : "{email}__{filename}"  — email is the unique person key.
Embeddings are injected directly — Chroma never re-embeds.

Search strategy:
  Fetch 2× candidates semantically, then re-rank:
    - Resumes containing the query keyword(s) literally → top
    - Rest → semantic order
  This handles specific tech queries ("hadoop", "langchain") accurately.
"""

import os
import re
import logging
from datetime import datetime, timezone

import chromadb

logger          = logging.getLogger(__name__)
CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "resumes"

_client     = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        logger.info(f"Initialising ChromaDB at '{CHROMA_PATH}'")
        _client     = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name     = COLLECTION_NAME,
            metadata = {"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB ready — {_collection.count()} resume(s)")
    return _collection


def _make_id(email: str, filename: str) -> str:
    safe_email    = re.sub(r"[^a-zA-Z0-9._\-]", "_", email)
    safe_filename = re.sub(r"[^a-zA-Z0-9._\-]", "_", filename)
    return f"{safe_email}__{safe_filename}"


def _keyword_boost(candidates: list, query: str) -> list:
    """
    Re-rank candidates so that resumes literally containing query
    keywords appear first. Within each group, semantic order is preserved.

    Why: cosine similarity on full-text embeddings can rank a resume about
    'distributed systems' above one explicitly mentioning 'hadoop', because
    they're semantically close. For specific tech keywords a recruiter types,
    literal presence should always win.
    """
    # Extract meaningful words from the query (ignore stopwords)
    stopwords = {"a", "an", "the", "and", "or", "for", "with", "who", "has",
                 "have", "in", "of", "to", "is", "are", "me", "find", "show",
                 "candidates", "person", "people", "someone", "skills", "skill"}
    query_words = [
        w.lower() for w in re.findall(r'\w+', query)
        if w.lower() not in stopwords and len(w) > 2
    ]

    if not query_words:
        return candidates

    keyword_hits = []
    no_hits      = []

    for c in candidates:
        doc_lower = (c.get("document", "") or "").lower()
        # Check if ANY query word appears literally in the stored resume text
        if any(word in doc_lower for word in query_words):
            keyword_hits.append(c)
        else:
            no_hits.append(c)

    logger.info(
        f"Keyword boost '{query}' → "
        f"{len(keyword_hits)} literal hits, {len(no_hits)} semantic-only"
    )
    return keyword_hits + no_hits


# ── Write ─────────────────────────────────────────────────────────────────────

def add_resume(
    *,
    text: str,
    embedding: list,
    name: str,
    email: str,
    filename: str,
    tfidf_score: float,
    embedding_score: float,
    hybrid_score: float,
    matches_jd: bool,
    session_id: str,
) -> str:
    """Upsert one resume as one chunk. Returns the doc_id."""
    col    = _get_collection()
    doc_id = _make_id(email, filename)

    col.upsert(
        ids        = [doc_id],
        documents  = [text],
        embeddings = [embedding],
        metadatas  = [{
            "name":            name,
            "email":           email,
            "filename":        filename,
            "tfidf_score":     round(tfidf_score, 4),
            "embedding_score": round(embedding_score, 4),
            "hybrid_score":    round(hybrid_score, 4),
            "matches_jd":      1 if matches_jd else 0,
            "session_id":      session_id,
            "uploaded_at":     datetime.now(timezone.utc).isoformat(),
        }],
    )
    logger.info(f"Stored → '{doc_id}'  name='{name}'  email='{email}'")
    return doc_id


# ── Read ──────────────────────────────────────────────────────────────────────

def query_resumes(
    query_embedding: list,
    query_text: str,          # ← the raw query string for keyword boosting
    n_results: int = 5,
    only_matched: bool = False,
) -> list[dict]:
    """
    Semantic search + keyword re-ranking.

    1. Fetch min(n_results * 2, total) candidates from ChromaDB semantically
       (fetching 2× gives keyword boost enough candidates to promote from)
    2. Re-rank: literal keyword matches first, semantic order within groups
    3. Return top n_results
    """
    col   = _get_collection()
    total = col.count()
    if total == 0:
        return []

    where = {"matches_jd": 1} if only_matched else None
    # Fetch 2× so keyword boost has room to reorder
    fetch = min(n_results * 2, total)

    try:
        results = col.query(
            query_embeddings = [query_embedding],
            n_results        = fetch,
            where            = where,
            include          = ["metadatas", "distances", "documents"],  # ← documents included
        )
    except Exception as e:
        logger.error(f"ChromaDB query error: {e}")
        return []

    candidates = []
    for doc_id, meta, dist, doc in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["documents"][0],
    ):
        candidates.append({
            "doc_id":           doc_id,
            "name":             meta.get("name", "Unknown"),
            "email":            meta.get("email", "N/A"),
            "filename":         meta.get("filename", ""),
            "tfidf_score":      meta.get("tfidf_score", 0.0),
            "embedding_score":  meta.get("embedding_score", 0.0),
            "hybrid_score":     meta.get("hybrid_score", 0.0),
            "matches_jd":       bool(meta.get("matches_jd", 0)),
            "search_similarity": round(1 - dist, 4),
            "uploaded_at":      meta.get("uploaded_at", ""),
            "document":         doc,   # full resume text for keyword check
        })

    # Re-rank: literal keyword hits first
    reranked = _keyword_boost(candidates, query_text)

    # Strip internal document field before returning (not needed by UI)
    for c in reranked:
        c.pop("document", None)

    logger.info(f"Query '{query_text}' → returning top {min(n_results, len(reranked))}")
    return reranked[:n_results]


def get_all_resumes() -> list[dict]:
    """Fetch every stored resume — used by the Rankings page."""
    col = _get_collection()
    if col.count() == 0:
        return []

    results = col.get(include=["metadatas"])
    records = []
    for doc_id, meta in zip(results["ids"], results["metadatas"]):
        records.append({
            "doc_id":          doc_id,
            "name":            meta.get("name", "Unknown"),
            "email":           meta.get("email", "N/A"),
            "filename":        meta.get("filename", ""),
            "tfidf_score":     meta.get("tfidf_score", 0.0),
            "embedding_score": meta.get("embedding_score", 0.0),
            "hybrid_score":    meta.get("hybrid_score", 0.0),
            "matches_jd":      bool(meta.get("matches_jd", 0)),
            "uploaded_at":     meta.get("uploaded_at", ""),
        })

    logger.info(f"get_all_resumes → {len(records)} record(s)")
    return records


def get_stats() -> dict:
    col = _get_collection()
    return {
        "total":      col.count(),
        "collection": COLLECTION_NAME,
        "path":       CHROMA_PATH,
    }

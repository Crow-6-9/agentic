import os
import re
import requests
import logging
from typing import Dict, Any, Optional
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.7
TFIDF_WEIGHT         = 0.4
EMBEDDING_WEIGHT     = 0.6

def get_embedding(api_key: str, input_text: str, model: str) -> Optional[list]:
    url = ""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": input_text,
        "model": model
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except Exception as err:
        logger.error(f"Error occurred: {err}")
        return None

def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_metadata(text: str) -> dict:
    """
    Extract name and email from resume text.
    - Email: reliable regex pattern match
    - Name:  first non-empty line heuristic (most resumes lead with the candidate's name)
    """
    # --- Email ---
    email_match = re.search(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        text
    )
    email = email_match.group(0) if email_match else "N/A"

    # --- Name ---
    # Take the first non-empty line; validate it looks like a real name
    # (2–4 words, only alpha + spaces/hyphens/apostrophes, 2–50 chars)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    raw_name = lines[0] if lines else "Unknown"
    name = raw_name if re.match(r"^[A-Za-z .'\-]{2,50}$", raw_name) else "Unknown"

    return {"name": name, "email": email}

def compute_tfidf_score(jd_text: str, resume_text: str) -> float:
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
        return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    except Exception as err:
        logger.error(f"TF-IDF error: {err}")
        return 0.0

def compute_hybrid_score(tfidf_score: float, embedding_score: float) -> float:
    return round(TFIDF_WEIGHT * tfidf_score + EMBEDDING_WEIGHT * embedding_score, 4)

def score_label(score: float) -> str:
    if score >= 0.85:
        return "🟢 Strong"
    elif score >= SIMILARITY_THRESHOLD:
        return "🟡 Good"
    else:
        return "🔴 Weak"

def build_results_table(sorted_resumes: list) -> str:
    header = (
        "| Rank | Name | Email | TF-IDF | Embedding | Hybrid | Fit |\n"
        "|:----:|------|-------|:------:|:---------:|:------:|:---:|\n"
    )
    rows = ""
    for i, r in enumerate(sorted_resumes, 1):
        rows += (
            f"| {i} "
            f"| {r['metadata']['name']} "
            f"| {r['metadata']['email']} "
            f"| {r['tfidf_score']:.4f} "
            f"| {r['embedding_score']:.4f} "
            f"| {r['hybrid_score']:.4f} "
            f"| {score_label(r['hybrid_score'])} |\n"
        )
    return header + rows

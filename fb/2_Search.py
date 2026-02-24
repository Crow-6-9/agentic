"""
Page 2 — Recruiter Search
Type a skill / query → embed → ChromaDB semantic search → results table.
"""

import logging

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

from services.embedding   import get_embedding
from services.matcher     import fit_label
from services.vectorstore import query_resumes, get_stats

st.set_page_config(page_title="Recruiter Search", page_icon="🔍", layout="wide")
st.title("🔍 Recruiter Search")
st.caption("Semantically search all stored resumes using natural language.")

# ── DB status ─────────────────────────────────────────────────────────────────
stats = get_stats()
if stats["total"] == 0:
    st.warning(
        "⚠️ No resumes in the database yet. "
        "Go to **📋 Matcher** first, upload a JD and some resumes."
    )
    st.stop()

st.info(f"💾 Searching across **{stats['total']}** stored resume(s).")

# ── Search controls ───────────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "Search query",
        placeholder = "e.g. 'LangChain developer', 'React and Node.js', 'ML engineer with Python'",
        label_visibility = "collapsed",
    )
with col2:
    n_results = st.number_input("Results", min_value=1, max_value=20, value=5, step=1)

only_matched = st.checkbox("Show only JD-matched candidates", value=False)

search_clicked = st.button("🔍 Search", use_container_width=True, type="primary")

# ── Search execution ──────────────────────────────────────────────────────────
if search_clicked:
    if not query.strip():
        st.warning("Please enter a search query.")
        st.stop()

    with st.spinner(f"Searching for: *{query}*..."):
        query_embedding = get_embedding(query)

    if not query_embedding:
        st.error("❌ Embedding API failed. Check your `.env` settings.")
        st.stop()

    with st.spinner("Querying ChromaDB..."):
        candidates = query_resumes(
            query_embedding = query_embedding,
            n_results       = int(n_results),
            only_matched    = only_matched,
        )

    logger.info(f"Search '{query}' → {len(candidates)} result(s)")

    if not candidates:
        st.warning(
            f"No results found for **\"{query}\"**. "
            "Try broader terms or uncheck 'JD-matched only'."
        )
        st.stop()

    # ── Summary metrics ───────────────────────────────────────────────────────
    top = candidates[0]
    st.success(
        f"**Top match:** {top['name']}  ·  📧 {top['email']}  ·  "
        f"Query similarity: `{top['search_similarity']:.4f}`"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Results returned", len(candidates))
    c2.metric("Top query match",  f"{top['search_similarity']:.4f}")
    c3.metric("Top JD hybrid",    f"{top['hybrid_score']:.4f}")

    # ── Results table ─────────────────────────────────────────────────────────
    def _fit_badge(label: str) -> str:
        return {"Strong": "🟢 Strong", "Good": "🟡 Good", "Weak": "🔴 Weak"}.get(label, label)

    rows = []
    for i, c in enumerate(candidates, 1):
        label = fit_label(c["hybrid_score"])
        rows.append({
            "Rank":          i,
            "Name":          c["name"],
            "Email":         c["email"],
            "File":          c["filename"],
            "Query Match":   round(c["search_similarity"], 4),
            "TF-IDF":        round(c["tfidf_score"], 4),
            "Embedding":     round(c["embedding_score"], 4),
            "Hybrid":        round(c["hybrid_score"], 4),
            "JD Match":      "✅" if c["matches_jd"] else "❌",
            "Fit":           _fit_badge(label),
            "Uploaded":      c["uploaded_at"][:10] if c.get("uploaded_at") else "N/A",
        })

    st.dataframe(
        rows,
        use_container_width = True,
        hide_index          = True,
        column_config       = {
            "Query Match": st.column_config.NumberColumn(format="%.4f"),
            "TF-IDF":      st.column_config.NumberColumn(format="%.4f"),
            "Embedding":   st.column_config.NumberColumn(format="%.4f"),
            "Hybrid":      st.column_config.NumberColumn(format="%.4f"),
        },
    )

    st.caption(
        "_Query Match = how well this resume matches your search query · "
        "TF-IDF/Embedding/Hybrid = scores from the original JD matching_"
    )

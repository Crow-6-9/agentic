"""
Page 1 — Resume Matcher
Upload JD + resumes → auto-process → score → store to ChromaDB.

Resumes are embedded using full text — same as Chainlit. Consistent vector space = accurate search.
This keeps one resume = one chunk, but the chunk is skill-signal-dense,
making semantic search (e.g. 'hadoop skills') accurate.
"""

import os
import uuid
import shutil
import tempfile
import logging

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

from services.parser      import extract_text_from_docx, extract_metadata
from services.embedding   import get_embedding
from services.matcher     import (
    compute_tfidf_score,
    compute_embedding_score,
    compute_hybrid_score,
    fit_label,
    is_match,
    TFIDF_WEIGHT,
    EMBEDDING_WEIGHT,
    SIMILARITY_THRESHOLD,
)
from services.vectorstore import add_resume, get_stats

st.set_page_config(page_title="Matcher", page_icon="📋", layout="wide")
st.title("📋 Resume Matcher")
st.caption(
    f"Hybrid scoring: TF-IDF `{int(TFIDF_WEIGHT*100)}%` + Embedding `{int(EMBEDDING_WEIGHT*100)}%`  ·  "
    f"Match threshold: `{SIMILARITY_THRESHOLD}`"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fit_badge(label: str) -> str:
    return {"Strong": "🟢 Strong", "Good": "🟡 Good", "Weak": "🔴 Weak"}.get(label, label)


def _save_resume_locally(tmp_path: str, filename: str) -> None:
    resumes_dir = os.path.join(os.path.dirname(__file__), "..", "resumes")
    os.makedirs(resumes_dir, exist_ok=True)
    shutil.copy2(tmp_path, os.path.join(resumes_dir, filename))


def _write_tmp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


# ── Session state ─────────────────────────────────────────────────────────────

if "jd_text"      not in st.session_state: st.session_state.jd_text      = None
if "jd_embedding" not in st.session_state: st.session_state.jd_embedding = None
if "session_id"   not in st.session_state: st.session_state.session_id   = str(uuid.uuid4())
if "results"      not in st.session_state: st.session_state.results       = []
if "processing"   not in st.session_state: st.session_state.processing    = False


# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — JOB DESCRIPTION
# ════════════════════════════════════════════════════════════════════════════════

st.subheader("Step 1 — Upload Job Description")

jd_file = st.file_uploader(
    "Upload the JD as a .docx file",
    type = ["docx"],
    key  = "jd_uploader",
)

if jd_file and not st.session_state.jd_text:
    # Auto-process JD as soon as it's uploaded — no button needed
    with st.spinner(f"Reading and embedding **{jd_file.name}**..."):
        tmp_path = _write_tmp(jd_file)
        try:
            jd_text = extract_text_from_docx(tmp_path)
        finally:
            os.unlink(tmp_path)

        jd_embedding = get_embedding(jd_text)

    if not jd_embedding:
        st.error("❌ Embedding failed. Check `API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_URL` in `.env`.")
    else:
        st.session_state.jd_text      = jd_text
        st.session_state.jd_embedding = jd_embedding
        st.session_state.results      = []
        logger.info(f"JD auto-loaded — session={st.session_state.session_id}  file={jd_file.name}")

if st.session_state.jd_text:
    st.success(f"✅ Job Description ready")
    with st.expander("Preview JD text"):
        st.text(st.session_state.jd_text[:1500] + ("..." if len(st.session_state.jd_text) > 1500 else ""))


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — RESUMES
# ════════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("Step 2 — Upload Resumes")

if not st.session_state.jd_text:
    st.info("⬆️ Upload a Job Description above first.")
else:
    resume_files = st.file_uploader(
        "Upload resumes (.docx) — up to 10 at once",
        type                  = ["docx"],
        accept_multiple_files = True,
        key                   = "resume_uploader",
    )

    if resume_files:
        # Auto-process as soon as files are uploaded — no button needed
        already_processed = {r["File"] for r in st.session_state.results}
        new_files = [f for f in resume_files if f.name not in already_processed]

        if new_files:
            if len(new_files) > 10:
                new_files = new_files[:10]

            jd_text      = st.session_state.jd_text
            jd_embedding = st.session_state.jd_embedding
            session_id   = st.session_state.session_id
            new_results  = []
            skipped      = 0

            progress = st.progress(0, text="Processing resumes...")

            for i, file in enumerate(new_files):
                progress.progress(i / len(new_files), text=f"Processing {file.name}...")

                tmp_path = _write_tmp(file)
                try:
                    full_text = extract_text_from_docx(tmp_path)
                    _save_resume_locally(tmp_path, file.name)
                except Exception as e:
                    st.warning(f"⚠️ Skipped **{file.name}** — could not read: {e}")
                    skipped += 1
                    os.unlink(tmp_path)
                    continue
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                meta = extract_metadata(full_text)

                # Embed full resume text — same approach as Chainlit.
                # Both resume and query are plain text → same vector space → accurate search.
                resume_embedding = get_embedding(full_text)

                if not resume_embedding:
                    st.warning(f"⚠️ Skipped **{file.name}** — embedding API failed.")
                    skipped += 1
                    continue

                tfidf   = compute_tfidf_score(jd_text, full_text)
                emb     = compute_embedding_score(jd_embedding, resume_embedding)
                hybrid  = compute_hybrid_score(tfidf, emb)
                matched = is_match(hybrid)
                label   = fit_label(hybrid)

                doc_id = add_resume(
                    text            = full_text,   # ← full text stored
                    embedding       = resume_embedding,
                    name            = meta["name"],
                    email           = meta["email"],
                    filename        = file.name,
                    tfidf_score     = tfidf,
                    embedding_score = emb,
                    hybrid_score    = hybrid,
                    matches_jd      = matched,
                    session_id      = session_id,
                )

                new_results.append({
                    "Name":      meta["name"],
                    "Email":     meta["email"],
                    "File":      file.name,
                    "TF-IDF":    round(tfidf, 4),
                    "Embedding": round(emb, 4),
                    "Hybrid":    round(hybrid, 4),
                    "Fit":       _fit_badge(label),
                    "Matches JD": "✅" if matched else "❌",
                    "_hybrid":   hybrid,
                    "_doc_id":   doc_id,
                })

                logger.info(
                    f"{meta['name']} ({meta['email']}) | "
                    f"tfidf={tfidf:.4f} emb={emb:.4f} hybrid={hybrid:.4f} → {label}"
                )

            progress.progress(1.0, text="Done!")
            st.session_state.results = st.session_state.results + new_results

            if skipped:
                st.warning(f"⚠️ {skipped} file(s) skipped.")

            stats = get_stats()
            st.success(
                f"✅ {len(new_results)} resume(s) processed and stored. "
                f"ChromaDB now holds **{stats['total']}** total resume(s)."
            )


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — RESULTS TABLE
# ════════════════════════════════════════════════════════════════════════════════

if st.session_state.results:
    st.divider()
    st.subheader("Step 3 — Ranked Results")

    sorted_results = sorted(st.session_state.results, key=lambda x: x["_hybrid"], reverse=True)
    display = [{k: v for k, v in r.items() if not k.startswith("_")} for r in sorted_results]

    matched   = sum(1 for r in sorted_results if r["Matches JD"] == "✅")
    unmatched = len(sorted_results) - matched

    col1, col2, col3 = st.columns(3)
    col1.metric("Total (this session)", len(sorted_results))
    col2.metric("🟢/🟡 Shortlisted",    matched)
    col3.metric("🔴 Below threshold",   unmatched)

    st.dataframe(
        display,
        use_container_width = True,
        hide_index          = True,
        column_config       = {
            "TF-IDF":    st.column_config.NumberColumn(format="%.4f"),
            "Embedding": st.column_config.NumberColumn(format="%.4f"),
            "Hybrid":    st.column_config.NumberColumn(format="%.4f"),
        },
    )

    if st.button("🔄 Start over (new JD)"):
        st.session_state.jd_text      = None
        st.session_state.jd_embedding = None
        st.session_state.results      = []
        st.session_state.session_id   = str(uuid.uuid4())
        st.rerun()

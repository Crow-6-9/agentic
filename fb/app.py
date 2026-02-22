import os
import sys
from pathlib import Path

# Add the root project directory to the Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import chainlit as cl
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the single .env file from the root
load_dotenv(dotenv_path=root_dir / ".env")

from back.core import (
    get_embedding, extract_text_from_docx, extract_metadata,
    compute_tfidf_score, compute_hybrid_score, score_label, build_results_table,
    SIMILARITY_THRESHOLD, TFIDF_WEIGHT, EMBEDDING_WEIGHT
)

# Environment variables
API_KEY = os.getenv("API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")          

async def step_upload_jd():
    """Step 1 — Prompt the user to upload the JD and process it."""
    files = await cl.AskFileMessage(
        content=(
            "### 📋 Step 1 of 3 — Upload Job Description\n"
            "Upload the Job Description as a **.docx** file to get started."
        ),
        accept=["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
        max_files=1,
        raise_on_timeout=False,
    ).send()

    if not files:
        await cl.Message(content="⚠️ No file received. Refresh and try again.").send()
        return

    file = files[0]
    jd_text      = extract_text_from_docx(file.path)
    jd_embedding = get_embedding(API_KEY, jd_text, EMBEDDING_MODEL)

    if not jd_embedding:
        await cl.Message(
            content="❌ Could not generate an embedding for the JD. Check your API key/endpoint."
        ).send()
        return

    cl.user_session.set("jd", jd_text)
    cl.user_session.set("jd_embedding", jd_embedding)
    await cl.Message(content=f"✅ **Job Description loaded:** {file.name}").send()

    await step_upload_resumes()   # auto-advance


async def step_upload_resumes():
    """Step 2 — Accept resumes and compute scores for each."""
    files = await cl.AskFileMessage(
        content=(
            "### 📂 Step 2 of 3 — Upload Resumes\n"
            "Upload up to **10 resumes** (.docx). "
            "Each one will be scored against the JD right away."
        ),
        accept=["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
        max_files=10,
        raise_on_timeout=False,
    ).send()

    if not files:
        await cl.Message(content="⚠️ No resumes received.").send()
        return

    jd_text      = cl.user_session.get("jd")
    jd_embedding = cl.user_session.get("jd_embedding")
    resumes      = cl.user_session.get("resumes")

    await cl.Message(content=f"⏳ Processing **{len(files)}** resume(s)...").send()

    for file in files:
        resume_text      = extract_text_from_docx(file.path)
        metadata         = extract_metadata(resume_text)
        resume_embedding = get_embedding(API_KEY, resume_text, EMBEDDING_MODEL)

        if not resume_embedding:
            await cl.Message(content=f"⚠️ Skipped **{file.name}** — embedding failed.").send()
            continue

        tfidf_score     = compute_tfidf_score(jd_text, resume_text)
        embedding_score = float(cosine_similarity([jd_embedding], [resume_embedding])[0][0])
        hybrid_score    = compute_hybrid_score(tfidf_score, embedding_score)

        resumes.append({
            "filename":        file.name,
            "metadata":        metadata,
            "tfidf_score":     tfidf_score,
            "embedding_score": embedding_score,
            "hybrid_score":    hybrid_score,
            "matches_jd":      hybrid_score >= SIMILARITY_THRESHOLD,
        })

        await cl.Message(
            content=(
                f"📄 **{file.name}**\n"
                f"> 👤 {metadata['name']}  ·  📧 {metadata['email']}\n"
                f"> TF-IDF: `{tfidf_score:.4f}` · Embedding: `{embedding_score:.4f}` "
                f"· Hybrid: `{hybrid_score:.4f}`  {score_label(hybrid_score)}"
            )
        ).send()

    cl.user_session.set("resumes", resumes)
    await step_show_results()    # auto-advance


async def step_show_results():
    """Step 3 — Display the ranked results table and offer next actions."""
    resumes = cl.user_session.get("resumes")

    if not resumes:
        await cl.Message(content="⚠️ No resumes to rank yet.").send()
        return

    sorted_resumes = sorted(resumes, key=lambda x: x["hybrid_score"], reverse=True)
    table          = build_results_table(sorted_resumes)
    matched        = sum(1 for r in sorted_resumes if r["matches_jd"])
    unmatched      = len(sorted_resumes) - matched

    await cl.Message(
        content=(
            "### 🏆 Step 3 of 3 — Ranked Results\n\n"
            f"**Scoring weights:** TF-IDF `{TFIDF_WEIGHT*100:.0f}%` + "
            f"Embedding `{EMBEDDING_WEIGHT*100:.0f}%`  ·  "
            f"**Match threshold:** `{SIMILARITY_THRESHOLD}`\n\n"
            f"{table}\n\n"
            f"**Summary:** 🟢/🟡 `{matched}` shortlisted  ·  🔴 `{unmatched}` below threshold\n\n"
            "_TF-IDF = keyword overlap · Embedding = semantic similarity · Hybrid = weighted blend_"
        )
    ).send()

    await cl.Message(
        content=(
            "**What would you like to do next?**\n"
            "- Say **`more resumes`** — upload additional resumes against the same JD\n"
            "- Say **`restart`** — start over with a new Job Description\n"
            "- Say **`show results`** — re-display the current rankings"
        )
    ).send()

@cl.on_chat_start
async def start():
    cl.user_session.set("resumes", [])
    cl.user_session.set("jd", None)
    cl.user_session.set("jd_embedding", None)

    await cl.Message(
        content=(
            "# 🤖 ATS Resume Matcher\n"
            "Hybrid scoring using **TF-IDF** (keyword overlap) + **Embeddings** (semantic meaning) "
            "— the same approach used by real ATS systems.\n\n"
            "Let's go! 👇"
        )
    ).send()

    await step_upload_jd()   # kick off the guided flow immediately

@cl.on_message
async def on_message(message: cl.Message):
    """Handle post-flow conversational commands."""
    text = message.content.strip().lower()

    if any(k in text for k in ["more resumes", "upload more", "add resumes"]):
        if not cl.user_session.get("jd"):
            await cl.Message(content="⚠️ No JD loaded. Say **restart** to begin again.").send()
        else:
            await step_upload_resumes()

    elif any(k in text for k in ["restart", "reset", "new jd", "start over"]):
        cl.user_session.set("resumes", [])
        cl.user_session.set("jd", None)
        cl.user_session.set("jd_embedding", None)
        await cl.Message(content="🔄 Session cleared. Starting fresh...").send()
        await step_upload_jd()

    elif any(k in text for k in ["result", "rank", "show", "table"]):
        await step_show_results()

    else:
        await cl.Message(
            content=(
                "I didn't catch that. You can say:\n"
                "- **`more resumes`** — upload more resumes against the current JD\n"
                "- **`show results`** — re-display the rankings\n"
                "- **`restart`** — start over with a new JD"
            )
        ).send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)

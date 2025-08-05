import streamlit as st
import asyncio
from graph_agent import create_graph_agent

agent = create_graph_agent()

st.set_page_config(page_title="Python Agent", layout="wide")
st.title("Agent Python 🐍")

# ✅ CSS Styling
st.markdown("""
    <style>
    .input-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        margin-top: 1rem;
    }
    .output-box {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        overflow-x: auto;
        margin-top: 1rem;
        white-space: pre-wrap;
        word-wrap: break-word;
        color: white;
    }
    .stTextArea textarea {
        height: 120px !important;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Layout block
with st.container():
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)

    query = st.text_area("💬 Ask a Python question or paste code:", placeholder="e.g. What does list comprehension mean?")
    run_clicked = st.button("▶ Run")
    
    st.markdown("</div>", unsafe_allow_html=True)

    output_placeholder = st.empty()

    async def run_agent(query: str):
        result = await agent.ainvoke({"input": query})
        return result.get("output", "").strip()

    if run_clicked and query.strip():
        with st.spinner("🔄 Thinking..."):
            output = asyncio.run(run_agent(query))

        # 🧠 Output display logic
        if output.startswith("Traceback") or "Error" in output:
            output_placeholder.error("⚠️ Error during code execution:")
            output_placeholder.code(output, language="python")
        elif any(line.strip().startswith(("def ", "class ", "import ", "print", "#")) for line in output.splitlines()):
            output_placeholder.code(output, language="python")
        else:
            output_placeholder.markdown(f"<div class='output-box'>{output}</div>", unsafe_allow_html=True)

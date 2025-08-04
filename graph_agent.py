import os
from langgraph.graph import StateGraph, END
from rag_filter import PythonRAGFilter             # ✅ RAG filter module
from llm.python_only_llm import PythonOnlyLLM      # ✅ Your custom LLM class

# ✅ Initialize LLM and RAG filter
llm = PythonOnlyLLM()
rag = PythonRAGFilter()  # loads vector DB from python_docs.txt

# ✅ LangGraph state object
class AgentState(dict):
    input: str
    output: str

# ✅ Main agent logic
def run_tool(state: AgentState) -> AgentState:
    user_input = state["input"]

    # 🔐 Use RAG to allow only Python questions
    if rag.is_python_question(user_input):
        context = rag.retrieve_context(user_input)

        prompt = f"""
You are a helpful Python assistant.
Answer the following question using this context:

----------------------
{context}
----------------------

Question: {user_input}
"""

        response = llm.invoke(prompt)
        return {
            "input": user_input,
            "output": response if isinstance(response, str) else response.content.strip()
        }

    else:
        return {
            "input": user_input,
            "output": "❌ Only Python-related questions are allowed. Please ask something related to Python programming."
        }

# ✅ LangGraph structure
def create_graph_agent():
    builder = StateGraph(AgentState)
    builder.add_node("process", run_tool)
    builder.set_entry_point("process")
    builder.add_edge("process", END)
    return builder.compile()

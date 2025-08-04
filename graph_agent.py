import os
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llm.python_only_llm import PythonOnlyLLM  # ✅ Use your custom class

# ✅ Instantiate your Python-only LLM wrapper
llm = PythonOnlyLLM()

# ✅ Load sentence-transformers model for intent classification
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Intent examples for similarity comparison
python_intents = [
    "How to create a list in Python?",
    "What is a Python dictionary?",
    "Write a Python function to reverse a string.",
    "Explain Python decorators.",
    "def greet(name): return f'Hello {name}'"
]

non_python_intents = [
    "What is the capital of India?",
    "Who is the prime minister?",
    "What is the time now?",
    "Play a song.",
    "Tell me a joke."
]

# ✅ Embedding similarity function
def is_python_intent(user_input: str, threshold: float = 0.6) -> bool:
    user_embedding = embedding_model.encode([user_input], convert_to_tensor=True)
    python_embeddings = embedding_model.encode(python_intents, convert_to_tensor=True)
    non_python_embeddings = embedding_model.encode(non_python_intents, convert_to_tensor=True)

    sim_with_python = cosine_similarity(user_embedding, python_embeddings).max()
    sim_with_non_python = cosine_similarity(user_embedding, non_python_embeddings).max()

    return sim_with_python > sim_with_non_python and sim_with_python > threshold

# ✅ Extra keyword-based check
def is_python_related(text: str) -> bool:
    keywords = [
        "python", "list", "tuple", "dictionary", "loop", "function", "class",
        "lambda", "comprehension", "decorator", "pandas", "numpy", "code",
        "def", "import", "syntax", "error", "exception", "recursion"
    ]
    return any(kw in text.lower() for kw in keywords)

# ✅ State object (LangGraph requirement)
class AgentState(dict):
    input: str
    output: str

# ✅ Core executor node
def run_tool(state: AgentState) -> AgentState:
    user_input = state["input"]

    # 🔐 Enforce Python-only logic
    if is_python_intent(user_input) or is_python_related(user_input):
        response = llm.invoke(user_input)
        return {
            "input": user_input,
            "output": response if isinstance(response, str) else response.content.strip()
        }
    else:
        return {
            "input": user_input,
            "output": "❌ This tool only handles Python-related queries or executable Python code."
        }

# ✅ LangGraph definition
def create_graph_agent():
    builder = StateGraph(AgentState)
    builder.add_node("process", run_tool)
    builder.set_entry_point("process")
    builder.add_edge("process", END)
    return builder.compile()

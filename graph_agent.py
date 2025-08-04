import os
from langgraph.graph import StateGraph, END
from llm.python_only_llm import PythonOnlyLLM  # ✅ Your custom LLM class

# ✅ Initialize LLM
llm = PythonOnlyLLM()

# ✅ LLM-based classifier to allow only Python-related queries
def is_python_question_llm(prompt: str) -> bool:
    judge_prompt = (
        "You are a classifier.\n\n"
        "Decide whether the following user question is related to Python programming or not.\n\n"
        f"Question: \"{prompt}\"\n\n"
        "Respond with just \"yes\" or \"no\"."
    )
    result = llm.invoke(judge_prompt).lower()
    return "yes" in result
  
# ✅ LangGraph state object
class AgentState(dict):
    input: str
    output: str

# ✅ Core logic: allow only Python-related questions
def run_tool(state: AgentState) -> AgentState:
    user_input = state["input"]

    if is_python_question_llm(user_input):
        response = llm.invoke(user_input)
        return {
            "input": user_input,
            "output": response if isinstance(response, str) else response.content.strip()
        }
    else:
        return {
            "input": user_input,
            "output": "❌ Only Python-related questions are allowed. Please ask something related to Python programming."
        }

# ✅ LangGraph pipeline
def create_graph_agent():
    builder = StateGraph(AgentState)
    builder.add_node("process", run_tool)
    builder.set_entry_point("process")
    builder.add_edge("process", END)
    return builder.compile()

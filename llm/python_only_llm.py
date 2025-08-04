import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

class PythonOnlyLLM:
    def __init__(self, temperature=0.2):
        self.llm = AzureChatOpenAI(
            azure_deployment=st.secrets["AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"],
            azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
            api_version=st.secrets["AZURE_OPENAI_API_VERSION"],
            temperature=temperature,
        )

        # Languages you want to block
        self.prog_langs = [
            "java", "c++", "c#", "javascript", "typescript", "go", "ruby", "rust",
            "swift", "kotlin", "perl", "php", "haskell", "scala", "r ", "matlab",
            "sql", "bash", "shell", "powershell", "vb.net", "dart"
        ]

    def _is_python_query(self, prompt: str) -> bool:
        prompt_lower = prompt.lower()

        # BLOCK if any other language is present
        if any(lang in prompt_lower for lang in self.prog_langs):
            return False
        
        # ALLOW only if "python" or a clearly standalone "py" appears
        return (
            "python" in prompt_lower
            or " py " in f" {prompt_lower} "
            or prompt_lower.strip().startswith("py")
        )

    def invoke(self, prompt: str):
        if not self._is_python_query(prompt):
            return "❌ This assistant only supports **Python-related** queries. Please ask something about Python."
        return self.llm.invoke(prompt)

    async def ainvoke(self, messages):
        if isinstance(messages, dict):
            prompt = messages.get("input", "")
        elif isinstance(messages, list):
            prompt = messages[-1]["content"] if messages else ""
        else:
            prompt = str(messages)

        if not self._is_python_query(prompt):
            return {"output": "❌ This assistant only supports **Python-related** queries. Please ask something about Python."}

        return await self.llm.ainvoke(messages)

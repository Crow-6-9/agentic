import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import re

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
    
        # Join with word boundaries to avoid matching substrings like "decorator" → "r"
        if re.search(r"\b(" + "|".join(re.escape(lang.strip()) for lang in self.prog_langs) + r")\b", prompt_lower):
            return False
    
        keywords = [
                  "python", "py", "def", "class", "lambda", "import", "from", "as", "return",
            "if", "elif", "else", "while", "for", "break", "continue", "pass", "with",
            "try", "except", "finally", "raise", "assert", "yield", "global", "nonlocal",
            "del", "async", "await", "not", "and", "or", "is", "in",
            "int", "float", "bool", "str", "list", "tuple", "set", "dict", "bytes", "None",
            "print", "len", "input", "open", "enumerate", "zip", "map", "filter", "reduce",
            "sorted", "reversed", "sum", "min", "max", "any", "all", "eval", "exec",
            "isinstance", "super", "abs", "round", "divmod", "pow", "bin", "oct", "hex",
            "list comprehension", "dict comprehension", "set comprehension",
            "generator expression", "self", "__init__", "__str__", "__repr__",
            "method", "property", "@staticmethod", "@classmethod", "@property",
            "@dataclass", "functools", "pandas", "numpy", "matplotlib", "seaborn",
            "requests", "os", "sys", "re", "json", "math", "datetime", "time", "csv",
            "random", "collections", "itertools", "typing", "unittest", "pytest",
            "try except", "traceback", "ValueError", "TypeError", "IndexError",
            "KeyError", "IOError", "ZeroDivisionError", "with open", "os.path",
            "argparse", "pathlib", "syntax", "semantics", "pep8", "virtualenv",
            "requirements.txt", "pip", "venv", "conda", "deepcopy", "shallow copy",
            "recursion", "iteration", "generator", "iterator", "comprehension",
            "zip", "enumerate", "lambda", "scikit-learn", "tensorflow", "keras",
            "torch", "huggingface", "transformers"
        ]
    
        return any(kw in prompt_lower for kw in keywords)



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




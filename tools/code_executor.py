def execute_python_code(code: str) -> str:
    if not code.strip().startswith(("def", "print", "import", "#", "for", "while", "if")):
        return "This tool only executes Python code."
    
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return "Code executed successfully." 
    except Exception as e:
        return f"Execution error: {e}"

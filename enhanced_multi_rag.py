# Advanced Multi-Agent RAG System
# Additional features: Vector store caching, advanced error handling, metrics tracking

import os
import json
import pickle
from datetime import datetime
from typing import TypedDict, Annotated, List, Sequence, Dict
import chainlit as cl
from pathlib import Path

# ... (all previous imports remain the same)

# ============================================
# ENHANCED PDF PROCESSOR WITH CACHING
# ============================================

class EnhancedPDFProcessor:
    """Enhanced PDF processor with vector store caching."""
    
    def __init__(self, pdf_path: str, cache_dir: str = "./vector_cache", 
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        self.pdf_path = pdf_path
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None
        self.documents = []
        
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_path(self) -> str:
        """Generate cache file path based on PDF name and parameters."""
        pdf_name = Path(self.pdf_path).stem
        cache_key = f"{pdf_name}_{self.chunk_size}_{self.chunk_overlap}"
        return os.path.join(self.cache_dir, f"{cache_key}.faiss")
    
    def _cache_exists(self) -> bool:
        """Check if cached vector store exists."""
        cache_path = self._get_cache_path()
        return os.path.exists(cache_path) and os.path.exists(f"{cache_path}.pkl")
    
    def load_and_process(self, force_reload: bool = False) -> FAISS:
        """Load PDF and create/load vector store with caching."""
        
        cache_path = self._get_cache_path()
        
        # Try to load from cache
        if not force_reload and self._cache_exists():
            print(f"📦 Loading vector store from cache: {cache_path}")
            embeddings = CapgeminiEmbeddings(
                api_key=API_KEY,
                base_url=OPENAI_COMPAT_URL,
                model=EMBEDDING_MODEL
            )
            self.vector_store = FAISS.load_local(
                cache_path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Vector store loaded from cache")
            return self.vector_store
        
        # Process PDF from scratch
        print(f"📄 Processing PDF: {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        raw_documents = loader.load()
        print(f"📖 Loaded {len(raw_documents)} pages")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.documents = text_splitter.split_documents(raw_documents)
        print(f"✂️  Split into {len(self.documents)} chunks")
        
        embeddings = CapgeminiEmbeddings(
            api_key=API_KEY,
            base_url=OPENAI_COMPAT_URL,
            model=EMBEDDING_MODEL
        )
        
        print("🔮 Creating embeddings and vector store...")
        self.vector_store = FAISS.from_documents(self.documents, embeddings)
        
        # Save to cache
        print(f"💾 Saving vector store to cache: {cache_path}")
        self.vector_store.save_local(cache_path)
        print("✅ Vector store cached successfully")
        
        return self.vector_store
    
    def get_retriever(self, k: int = 4):
        """Get a retriever from the vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load_and_process() first.")
        return self.vector_store.as_retriever(search_kwargs={"k": k})

# ============================================
# METRICS TRACKER
# ============================================

class MetricsTracker:
    """Track agent performance metrics."""
    
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "by_agent": {
                "conversational": 0,
                "factual_agent": 0,
                "analytical_agent": 0,
                "complex_agent": 0
            },
            "avg_response_time": 0,
            "response_times": []
        }
    
    def record_query(self, agent_type: str, response_time: float):
        """Record a query and its metrics."""
        self.metrics["total_queries"] += 1
        if agent_type in self.metrics["by_agent"]:
            self.metrics["by_agent"][agent_type] += 1
        
        self.metrics["response_times"].append(response_time)
        self.metrics["avg_response_time"] = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
    
    def get_summary(self) -> str:
        """Get formatted metrics summary."""
        total = self.metrics["total_queries"]
        if total == 0:
            return "No queries processed yet."
        
        summary = f"""
📊 **Session Metrics**

**Total Queries**: {total}

**Agent Usage**:
- Conversational: {self.metrics['by_agent']['conversational']} ({self.metrics['by_agent']['conversational']/total*100:.1f}%)
- Factual: {self.metrics['by_agent']['factual_agent']} ({self.metrics['by_agent']['factual_agent']/total*100:.1f}%)
- Analytical: {self.metrics['by_agent']['analytical_agent']} ({self.metrics['by_agent']['analytical_agent']/total*100:.1f}%)
- Complex: {self.metrics['by_agent']['complex_agent']} ({self.metrics['by_agent']['complex_agent']/total*100:.1f}%)

**Performance**:
- Avg Response Time: {self.metrics['avg_response_time']:.2f}s
        """
        return summary.strip()

# ============================================
# ENHANCED MULTI-AGENT CLASS
# ============================================

class EnhancedMultiAgentPDFQA:
    """Enhanced multi-agent system with metrics and better error handling."""
    
    def __init__(self, pdf_path: str, api_key: str = None, use_cache: bool = True):
        self.api_key = api_key or API_KEY
        
        if not self.api_key:
            raise ValueError("API_KEY is required.")
        
        self.llm = ChatOpenAI(
            model=COMPLETION_MODEL,
            temperature=0.3,
            openai_api_key=self.api_key,
            openai_api_base=OPENAI_COMPAT_URL,
        )
        
        # Use enhanced processor with caching
        self.pdf_processor = EnhancedPDFProcessor(pdf_path)
        self.vector_store = self.pdf_processor.load_and_process(force_reload=not use_cache)
        self.retriever = self.pdf_processor.get_retriever(k=4)
        
        self.app = build_multi_agent_graph(self.retriever, self.llm)
        self.messages: List[BaseMessage] = []
        self.metrics = MetricsTracker()
    
    def ask(self, question: str) -> tuple[str, List[str], str]:
        """Ask a question and return response with steps and agent type."""
        start_time = datetime.now()
        
        initial_state = {
            "messages": self.messages,
            "query": question,
            "context": "",
            "response": "",
            "needs_retrieval": True,
            "query_type": "",
            "route": "",
            "intermediate_steps": []
        }
        
        try:
            result = self.app.invoke(initial_state)
            
            self.messages.append(HumanMessage(content=question))
            self.messages.append(AIMessage(content=result["response"]))
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Get agent type from route
            agent_type = result.get("route", "unknown")
            
            # Record metrics
            self.metrics.record_query(agent_type, response_time)
            
            return result["response"], result.get("intermediate_steps", []), agent_type
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            return error_msg, [f"Error: {str(e)}"], "error"
    
    def clear_history(self):
        """Clear conversation history."""
        self.messages = []
    
    def get_metrics(self) -> str:
        """Get performance metrics."""
        return self.metrics.get_summary()

# ============================================
# ENHANCED CHAINLIT UI
# ============================================

@cl.on_chat_start
async def start():
    """Initialize the chat session with enhanced features."""
    
    # Show loading message
    init_msg = cl.Message(content="🚀 **Initializing Multi-Agent System...**")
    await init_msg.send()
    
    pdf_path = os.getenv("PDF_PATH", "your_document.pdf")
    
    try:
        # Initialize agent with progress updates
        init_msg.content = "📦 Loading PDF and creating embeddings..."
        await init_msg.update()
        
        agent = EnhancedMultiAgentPDFQA(pdf_path, use_cache=True)
        cl.user_session.set("agent", agent)
        
        init_msg.content = "✅ **System Ready!**"
        await init_msg.update()
        
        welcome_message = f"""
# 🤖 Multi-Agent PDF Q&A System

Welcome! I'm powered by **5 specialized AI agents** working together:

### 🎯 Agent Roles

| Agent | Purpose |
|-------|---------|
| 🧭 Router | Analyzes your question and routes to the right specialist |
| 💬 Conversational | Handles greetings and casual chat |
| 📊 Factual | Direct, accurate answers from the document |
| 🔍 Analytical | Deep analysis, comparisons, and insights |
| 🧠 Complex | Multi-step reasoning for complex questions |

### 📋 Document Info
- **File**: `{Path(pdf_path).name}`
- **Model**: `{COMPLETION_MODEL}`
- **Status**: ✅ Ready

### 💡 Tips
- Ask factual questions for quick answers
- Request analysis for comparisons or insights
- Use `/metrics` to see usage statistics
- Use `/clear` to reset conversation

**Ask me anything about the document!**
        """
        
        await cl.Message(content=welcome_message).send()
        
    except Exception as e:
        error_msg = f"""
❌ **Initialization Error**

```
{str(e)}
```

**Troubleshooting Steps:**
1. Verify PDF_PATH in your .env file
2. Check API credentials
3. Ensure all dependencies are installed
        """
        await cl.Message(content=error_msg).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages with enhanced features."""
    agent = cl.user_session.get("agent")
    
    if not agent:
        await cl.Message(
            content="⚠️ Agent not initialized. Please refresh the page."
        ).send()
        return
    
    # Handle special commands
    if message.content.strip().lower() == "/metrics":
        metrics = agent.get_metrics()
        await cl.Message(content=metrics).send()
        return
    
    if message.content.strip().lower() == "/clear":
        agent.clear_history()
        await cl.Message(content="🗑️ Conversation history cleared!").send()
        return
    
    if message.content.strip().lower() == "/help":
        help_msg = """
### 📚 Available Commands

- `/metrics` - View session statistics
- `/clear` - Clear conversation history
- `/help` - Show this help message

### 💬 How to Ask Questions

**Factual**: "What is the definition of X?"
**Analytical**: "Compare X and Y"
**Complex**: "Explain the process of..."
        """
        await cl.Message(content=help_msg).send()
        return
    
    # Show processing message
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Get response from multi-agent system
        response, steps, agent_type = agent.ask(message.content)
        
        # Determine agent emoji
        agent_emojis = {
            "conversational": "💬",
            "factual_agent": "📊",
            "analytical_agent": "🔍",
            "complex_agent": "🧠"
        }
        agent_emoji = agent_emojis.get(agent_type, "🤖")
        
        # Format workflow steps
        step_details = "\n".join([f"  {i+1}. {step}" for i, step in enumerate(steps)])
        
        final_message = f"""
{response}

---
{agent_emoji} **Processed by**: `{agent_type.replace('_', ' ').title()}`

<details>
<summary><b>🔄 Agent Workflow</b></summary>

{step_details}

</details>
        """
        
        msg.content = final_message
        await msg.update()
        
    except Exception as e:
        error_msg = f"""
❌ **Error Processing Query**

```
{str(e)}
```

Please try rephrasing your question or use `/help` for assistance.
        """
        msg.content = error_msg
        await msg.update()

@cl.on_chat_end
async def end():
    """Handle chat end with metrics summary."""
    agent = cl.user_session.get("agent")
    
    if agent:
        metrics = agent.get_metrics()
        goodbye_msg = f"""
👋 **Thanks for using the Multi-Agent PDF Q&A System!**

{metrics}

Have a great day! 🌟
        """
        await cl.Message(content=goodbye_msg).send()
    else:
        await cl.Message(content="👋 Goodbye!").send()

if __name__ == "__main__":
    pass

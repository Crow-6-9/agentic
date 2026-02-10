import os
import requests
from typing import TypedDict, Annotated, List, Sequence, Literal
from operator import add
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.embeddings.base import Embeddings
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

BASE_URL = os.getenv("BASE_URL")
OPENAI_COMPAT_URL = os.getenv("OPENAI_COMPAT_URL")
COMPLETION_MODEL = os.getenv("COMPLETION_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
API_KEY = os.getenv("API_KEY")

# ============================================
# STATE DEFINITION
# ============================================

class AgentState(TypedDict):
    """Enhanced state for multi-agent system."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    context: str
    response: str
    needs_retrieval: bool
    query_type: str  # 'greeting', 'factual', 'analytical', 'complex'
    route: str  # Which agent to use
    intermediate_steps: List[str]

# ============================================
# CUSTOM EMBEDDINGS
# ============================================

class CapgeminiEmbeddings(Embeddings):    
    def __init__(
        self, 
        api_key: str, 
        base_url: str, 
        model: str = "amazon.titan-embed-text-v2:0"
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
                
    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        endpoint = f"{self.base_url}/embeddings"
        payload = {
            "model": self.model,
            "input": texts
        }
        
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if "data" in result:
                    return [item["embedding"] for item in result["data"]]
                elif "embeddings" in result:
                    return result["embeddings"]
                else:
                    return result
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
                    
        except Exception as e:
            print(f"Failed with endpoint {endpoint}: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._get_embedding(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._get_embedding([text])[0]

# ============================================
# PDF PROCESSOR
# ============================================

class PDFProcessor:
    """Handles PDF loading, text splitting, and vector store creation."""
    
    def __init__(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None
        self.documents = []
        
    def load_and_process(self) -> FAISS:
        """Load PDF, split into chunks, and create vector store."""
        loader = PyPDFLoader(self.pdf_path)
        raw_documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.documents = text_splitter.split_documents(raw_documents)
        
        embeddings = CapgeminiEmbeddings(
            api_key=API_KEY,
            base_url=OPENAI_COMPAT_URL,
            model=EMBEDDING_MODEL
        )
        
        self.vector_store = FAISS.from_documents(self.documents, embeddings)
        
        return self.vector_store
    
    def get_retriever(self, k: int = 4):
        """Get a retriever from the vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load_and_process() first.")
        return self.vector_store.as_retriever(search_kwargs={"k": k})

# ============================================
# MULTI-AGENT NODES
# ============================================

def create_multi_agent_nodes(retriever, llm):
    """Create specialized agent nodes."""
    
    # 1. ROUTER AGENT - Determines query type and routes to appropriate agent
    def router_agent(state: AgentState) -> AgentState:
        """Analyze query and route to appropriate specialized agent."""
        query = state["query"]
        
        # Simple greeting detection
        greetings = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"]
        if query.lower().strip() in greetings:
            return {
                **state,
                "query_type": "greeting",
                "route": "conversational",
                "needs_retrieval": False,
                "intermediate_steps": state.get("intermediate_steps", []) + ["Router: Detected greeting"]
            }
        
        # Use LLM to classify query type
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query router. Classify the user query into ONE of these categories:
            - 'factual': Simple factual questions that need direct information lookup
            - 'analytical': Questions requiring analysis, comparison, or synthesis of information
            - 'complex': Multi-part questions or questions requiring step-by-step reasoning
            
            Respond with ONLY the category name, nothing else."""),
            ("human", "{query}")
        ])
        
        chain = router_prompt | llm
        classification = chain.invoke({"query": query}).content.strip().lower()
        
        # Map classification to route
        route_map = {
            "factual": "factual_agent",
            "analytical": "analytical_agent",
            "complex": "complex_agent"
        }
        
        route = route_map.get(classification, "factual_agent")
        
        return {
            **state,
            "query_type": classification,
            "route": route,
            "needs_retrieval": True,
            "intermediate_steps": state.get("intermediate_steps", []) + [f"Router: Classified as '{classification}', routing to '{route}'"]
        }
    
    # 2. RETRIEVAL AGENT - Retrieves relevant documents
    def retrieval_agent(state: AgentState) -> AgentState:
        """Retrieve relevant document chunks."""
        if not state.get("needs_retrieval", True):
            return state
            
        query = state["query"]
        query_type = state.get("query_type", "factual")
        
        # Adjust retrieval based on query type
        k = 4 if query_type == "factual" else 6 if query_type == "analytical" else 8
        
        docs = retriever.invoke(query)[:k]
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            page_num = doc.metadata.get("page", "Unknown")
            context_parts.append(f"[Section {i} - Page {page_num}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        return {
            **state,
            "context": context,
            "intermediate_steps": state.get("intermediate_steps", []) + [f"Retrieval: Found {len(docs)} relevant sections"]
        }
    
    # 3. CONVERSATIONAL AGENT - Handles greetings and casual conversation
    def conversational_agent(state: AgentState) -> AgentState:
        """Handle greetings and general conversation."""
        query = state["query"]
        messages = state.get("messages", [])
        
        system_prompt = """You are a friendly assistant specialized in Medical Device Software Lifecycle
        Risk Management procedures. You're here to help users understand the document.
        Keep responses warm and professional."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}")
        ])
        
        history = messages[-10:] if len(messages) > 10 else messages
        chain = prompt | llm
        response = chain.invoke({"history": history, "query": query})
        
        return {
            **state,
            "response": response.content,
            "intermediate_steps": state.get("intermediate_steps", []) + ["Conversational Agent: Generated response"],
            "messages": [HumanMessage(content=query), AIMessage(content=response.content)]
        }
    
    # 4. FACTUAL AGENT - Handles straightforward factual queries
    def factual_agent(state: AgentState) -> AgentState:
        """Handle factual queries with direct answers."""
        query = state["query"]
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        system_prompt = """You are a factual information specialist for Medical Device Software Lifecycle
        Risk Management procedures.
        
        Context from the document:
        {context}
        
        Provide direct, accurate answers based strictly on the context provided.
        If the information is not in the context, clearly state that."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}")
        ])
        
        history = messages[-10:] if len(messages) > 10 else messages
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "history": history,
            "query": query
        })
        
        return {
            **state,
            "response": response.content,
            "intermediate_steps": state.get("intermediate_steps", []) + ["Factual Agent: Generated response"],
            "messages": [HumanMessage(content=query), AIMessage(content=response.content)]
        }
    
    # 5. ANALYTICAL AGENT - Handles analysis and comparison queries
    def analytical_agent(state: AgentState) -> AgentState:
        """Handle analytical queries requiring synthesis."""
        query = state["query"]
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        system_prompt = """You are an analytical specialist for Medical Device Software Lifecycle
        Risk Management procedures.
        
        Context from the document:
        {context}
        
        Analyze the information, identify patterns, compare different aspects, and provide
        insightful synthesis. Structure your response clearly with reasoning."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}")
        ])
        
        history = messages[-10:] if len(messages) > 10 else messages
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "history": history,
            "query": query
        })
        
        return {
            **state,
            "response": response.content,
            "intermediate_steps": state.get("intermediate_steps", []) + ["Analytical Agent: Generated response"],
            "messages": [HumanMessage(content=query), AIMessage(content=response.content)]
        }
    
    # 6. COMPLEX AGENT - Handles multi-step reasoning
    def complex_agent(state: AgentState) -> AgentState:
        """Handle complex queries with step-by-step reasoning."""
        query = state["query"]
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        system_prompt = """You are a complex reasoning specialist for Medical Device Software Lifecycle
        Risk Management procedures.
        
        Context from the document:
        {context}
        
        Break down complex questions into steps. Provide thorough, step-by-step reasoning.
        Show your thought process clearly."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}")
        ])
        
        history = messages[-10:] if len(messages) > 10 else messages
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "history": history,
            "query": query
        })
        
        return {
            **state,
            "response": response.content,
            "intermediate_steps": state.get("intermediate_steps", []) + ["Complex Agent: Generated response"],
            "messages": [HumanMessage(content=query), AIMessage(content=response.content)]
        }
    
    return (router_agent, retrieval_agent, conversational_agent, 
            factual_agent, analytical_agent, complex_agent)

# ============================================
# CONDITIONAL ROUTING
# ============================================

def route_query(state: AgentState) -> str:
    """Route to appropriate agent based on classification."""
    return state.get("route", "factual_agent")

def should_retrieve(state: AgentState) -> str:
    """Determine if retrieval is needed."""
    if state.get("needs_retrieval", True):
        return "retrieve"
    return state.get("route", "conversational")

# ============================================
# BUILD MULTI-AGENT GRAPH
# ============================================

def build_multi_agent_graph(retriever, llm) -> StateGraph:
    """Build the multi-agent graph."""
    
    (router_agent, retrieval_agent, conversational_agent,
     factual_agent, analytical_agent, complex_agent) = create_multi_agent_nodes(retriever, llm)
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_agent)
    workflow.add_node("retrieve", retrieval_agent)
    workflow.add_node("conversational", conversational_agent)
    workflow.add_node("factual_agent", factual_agent)
    workflow.add_node("analytical_agent", analytical_agent)
    workflow.add_node("complex_agent", complex_agent)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        should_retrieve,
        {
            "retrieve": "retrieve",
            "conversational": "conversational"
        }
    )
    
    # Add conditional edges from retrieval to specific agents
    workflow.add_conditional_edges(
        "retrieve",
        route_query,
        {
            "factual_agent": "factual_agent",
            "analytical_agent": "analytical_agent",
            "complex_agent": "complex_agent"
        }
    )
    
    # All agents end
    workflow.add_edge("conversational", END)
    workflow.add_edge("factual_agent", END)
    workflow.add_edge("analytical_agent", END)
    workflow.add_edge("complex_agent", END)
    
    return workflow.compile()

# ============================================
# MAIN AGENT CLASS
# ============================================

class MultiAgentPDFQA:
    def __init__(self, pdf_path: str, api_key: str = None):
        self.api_key = api_key or API_KEY
        
        if not self.api_key:
            raise ValueError("API_KEY is required.")
        
        self.llm = ChatOpenAI(
            model=COMPLETION_MODEL,
            temperature=0.3,
            openai_api_key=self.api_key,
            openai_api_base=OPENAI_COMPAT_URL,
        )
        
        self.pdf_processor = PDFProcessor(pdf_path)
        self.vector_store = self.pdf_processor.load_and_process()
        self.retriever = self.pdf_processor.get_retriever(k=4)
        
        self.app = build_multi_agent_graph(self.retriever, self.llm)
        self.messages: List[BaseMessage] = []
    
    def ask(self, question: str) -> tuple[str, List[str]]:
        """Ask a question and return response with intermediate steps."""
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
        
        result = self.app.invoke(initial_state)
        
        self.messages.append(HumanMessage(content=question))
        self.messages.append(AIMessage(content=result["response"]))
        
        return result["response"], result.get("intermediate_steps", [])
    
    def clear_history(self):
        """Clear conversation history."""
        self.messages = []

# ============================================
# CHAINLIT UI
# ============================================

@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    await cl.Message(
        content="🔧 **Initializing Multi-Agent PDF Q&A System...**"
    ).send()
    
    # Get PDF path from environment or use default
    pdf_path = os.getenv("PDF_PATH", "your_document.pdf")
    
    try:
        agent = MultiAgentPDFQA(pdf_path)
        cl.user_session.set("agent", agent)
        
        welcome_message = f"""
# 🤖 Multi-Agent PDF Q&A System

Welcome! I'm powered by multiple specialized AI agents:

- **🧭 Router Agent**: Analyzes your question and routes it to the right specialist
- **💬 Conversational Agent**: Handles greetings and casual conversation
- **📊 Factual Agent**: Provides direct, factual answers
- **🔍 Analytical Agent**: Performs analysis and comparisons
- **🧠 Complex Agent**: Handles multi-step reasoning

**Document loaded**: `{pdf_path}`
**Model**: `{COMPLETION_MODEL}`

Ask me anything about the document!

*Type your question below to get started...*
        """
        
        await cl.Message(content=welcome_message).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ **Error initializing agent**: {str(e)}"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages."""
    agent = cl.user_session.get("agent")
    
    if not agent:
        await cl.Message(
            content="⚠️ Agent not initialized. Please refresh the page."
        ).send()
        return
    
    # Show processing message
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Get response from multi-agent system
        response, steps = agent.ask(message.content)
        
        # Create response with intermediate steps
        step_details = "\n".join([f"- {step}" for step in steps])
        
        final_message = f"""
{response}

---
**🔄 Agent Workflow:**
{step_details}
        """
        
        msg.content = final_message
        await msg.update()
        
    except Exception as e:
        msg.content = f"❌ **Error**: {str(e)}"
        await msg.update()

@cl.on_chat_end
async def end():
    """Handle chat end."""
    await cl.Message(content="👋 Thanks for using the Multi-Agent PDF Q&A System!").send()

# ============================================
# RUN CHAINLIT APP
# ============================================

if __name__ == "__main__":
    # Chainlit will automatically run when you execute: chainlit run multi_agent_rag.py
    pass

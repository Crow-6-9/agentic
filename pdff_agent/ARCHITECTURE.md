# Multi-Agent RAG System Architecture

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Chainlit UI Layer                         │
│  (User Interface, Message Handling, Session Management)         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Multi-Agent Orchestration                      │
│                      (LangGraph Workflow)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  Router Agent  │
                    │  (Classifier)  │
                    └────────┬───────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
      ┌──────────┐   ┌─────────────┐   ┌──────────┐
      │Greeting? │   │Need Context?│   │ Complex? │
      └────┬─────┘   └──────┬──────┘   └────┬─────┘
           │                │                │
           ▼                ▼                ▼
    ┌─────────────┐  ┌────────────┐  ┌─────────────┐
    │Conversational│  │ Retrieval  │  │  Retrieval  │
    │   Agent     │  │   Agent    │  │   Agent     │
    └─────┬───────┘  └─────┬──────┘  └──────┬──────┘
          │                │                 │
          │         ┌──────┴──────┐         │
          │         │             │         │
          │         ▼             ▼         ▼
          │  ┌──────────┐  ┌──────────┐  ┌──────────┐
          │  │ Factual  │  │Analytical│  │ Complex  │
          │  │  Agent   │  │  Agent   │  │  Agent   │
          │  └────┬─────┘  └────┬─────┘  └────┬─────┘
          │       │             │             │
          └───────┴─────────────┴─────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │    Response    │
                    │   Generation   │
                    └────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Support Services                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │Vector Store  │  │  Embeddings  │  │   Metrics    │         │
│  │  (FAISS)     │  │ (Capgemini)  │  │   Tracker    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Data Flow

### 1. Query Processing Flow

```
User Input → Router Agent → Query Classification
                                    │
                     ┌──────────────┼──────────────┐
                     │              │              │
                Greeting      Simple Fact    Complex Question
                     │              │              │
                     ▼              ▼              ▼
              Conversational   Factual Agt    Complex Agt
                     │              │              │
                     │         ┌────┴────┐         │
                     │         ▼         ▼         ▼
                     │    Retrieval  Retrieval  Retrieval
                     │    (4 chunks) (4-6)      (6-8)
                     │         │         │         │
                     └─────────┴─────────┴─────────┘
                                    │
                                    ▼
                            Response to User
```

### 2. Retrieval Flow

```
Query → Embedding Generation → Vector Similarity Search
                                        │
                                        ▼
                            Top-K Document Chunks
                                        │
                                        ▼
                            Context Assembly
                                        │
                                        ▼
                            LLM Prompt Construction
                                        │
                                        ▼
                            Response Generation
```

## 🎯 Agent Specifications

### Router Agent
**Purpose**: Query classification and routing
**Input**: User query string
**Output**: Query type + target agent route
**Logic**:
- Greeting detection (pattern matching)
- LLM-based classification for complex queries
- Routes to: conversational, factual_agent, analytical_agent, or complex_agent

### Conversational Agent
**Purpose**: Handle greetings and casual conversation
**Triggers**: "hi", "hello", "thanks", "bye", etc.
**Features**:
- No document retrieval
- Friendly, professional tone
- Quick responses

### Factual Agent
**Purpose**: Direct, accurate answers
**Retrieval**: 4 document chunks
**Style**: Concise, citation-ready
**Best for**: 
- "What is X?"
- "Define Y"
- "List the requirements for Z"

### Analytical Agent
**Purpose**: Analysis and synthesis
**Retrieval**: 4-6 document chunks
**Style**: Comparative, insightful
**Best for**:
- "Compare X and Y"
- "What are the differences between..."
- "Analyze the relationship..."

### Complex Agent
**Purpose**: Multi-step reasoning
**Retrieval**: 6-8 document chunks
**Style**: Step-by-step, detailed
**Best for**:
- "Explain the process of..."
- "How do I implement..."
- Multi-part questions

## 🔧 Technology Stack

### Core Framework
- **LangChain**: LLM orchestration and document processing
- **LangGraph**: Multi-agent workflow management
- **Chainlit**: Web UI and user interaction

### Vector Database
- **FAISS**: Fast similarity search and clustering
- **Custom Embeddings**: Capgemini/AWS Bedrock compatible

### Document Processing
- **PyPDF**: PDF parsing and text extraction
- **RecursiveCharacterTextSplitter**: Intelligent chunking

### LLM Integration
- **OpenAI-compatible API**: Flexible model support
- **Custom base URL**: Works with various providers

## 📁 File Structure

```
multi-agent-rag/
├── multi_agent_rag.py           # Main application
├── enhanced_multi_agent_rag.py  # Version with caching & metrics
├── requirements.txt              # Python dependencies
├── .env                         # Environment configuration
├── .env.example                 # Configuration template
├── .chainlit                    # Chainlit config
├── start.sh                     # Quick start script
├── README.md                    # Documentation
├── ARCHITECTURE.md              # This file
└── vector_cache/                # Cached vector stores
    └── [pdf_name].faiss
```

## 🔄 State Management

### AgentState Schema
```python
{
    "messages": List[BaseMessage],      # Conversation history
    "query": str,                       # Current user query
    "context": str,                     # Retrieved document context
    "response": str,                    # Generated response
    "needs_retrieval": bool,            # Whether to retrieve docs
    "query_type": str,                  # Classification result
    "route": str,                       # Target agent name
    "intermediate_steps": List[str]     # Workflow tracking
}
```

## 🚀 Performance Optimizations

### 1. Vector Store Caching
- Cache FAISS index to disk
- Avoid re-embedding on restart
- Hash-based cache invalidation

### 2. Adaptive Retrieval
- 4 chunks for factual queries
- 6 chunks for analytical queries
- 8 chunks for complex queries

### 3. Conversation History Management
- Keep last 10 messages for context
- Prevents token limit issues
- Maintains relevance

### 4. Batch Processing
- Process embeddings in batches of 10
- Reduces API calls
- Improves throughput

## 📈 Metrics Tracking

Tracks:
- Total queries processed
- Queries per agent type
- Average response time
- Agent usage distribution

## 🔐 Security Features

- Environment variable configuration
- API key protection
- Input validation
- Error handling with sanitized messages

## 🎨 UI Features

### Chainlit Integration
- Real-time message streaming
- Markdown rendering
- Code syntax highlighting
- Collapsible workflow details
- Session persistence
- Custom styling support

### Special Commands
- `/metrics` - View statistics
- `/clear` - Reset conversation
- `/help` - Show help

## 🔮 Future Enhancements

### Planned Features
1. **Agent Memory**: Cross-session knowledge retention
2. **Multi-document Support**: Query multiple PDFs
3. **Custom Agent Creation**: User-defined specialists
4. **Voice Integration**: Speech-to-text input
5. **Advanced Analytics**: Query pattern analysis
6. **Document Comparison**: Cross-reference capabilities
7. **Export Options**: Save conversations as PDF/Markdown
8. **Collaborative Features**: Multi-user sessions

### Scalability Considerations
- Redis for distributed caching
- Async processing for large documents
- Queue-based job management
- Load balancing for multiple users
- Database backend for persistence

## 📚 References

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [Chainlit Docs](https://docs.chainlit.io/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)

---

**Last Updated**: 2024
**Version**: 1.0.0

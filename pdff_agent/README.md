# 🤖 Multi-Agent RAG System with Chainlit UI

A sophisticated multi-agent Retrieval-Augmented Generation (RAG) system for intelligent PDF question answering, powered by LangGraph and featuring a beautiful Chainlit interface.

## 🌟 Features

### Multi-Agent Architecture
- **🧭 Router Agent**: Intelligently analyzes queries and routes them to specialized agents
- **💬 Conversational Agent**: Handles greetings and casual conversation
- **📊 Factual Agent**: Provides direct, accurate answers from the document
- **🔍 Analytical Agent**: Performs deep analysis, comparisons, and synthesis
- **🧠 Complex Agent**: Handles multi-step reasoning and complex queries

### Key Capabilities
- ✅ Intelligent query classification and routing
- ✅ Context-aware document retrieval
- ✅ Adaptive retrieval (4-8 chunks based on query complexity)
- ✅ Conversation history management
- ✅ Beautiful, interactive Chainlit UI
- ✅ Real-time agent workflow visualization
- ✅ Support for custom embeddings (Capgemini/AWS compatible)

## 📋 Prerequisites

- Python 3.9 or higher
- API access to your LLM provider
- PDF document to query

## 🚀 Installation

### 1. Clone or Download the Code

```bash
# Create project directory
mkdir multi-agent-rag
cd multi-agent-rag
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# API Configuration
API_KEY=your_actual_api_key
BASE_URL=https://your-base-url.com
OPENAI_COMPAT_URL=https://your-openai-compatible-endpoint.com

# Model Configuration
COMPLETION_MODEL=your-completion-model-name
EMBEDDING_MODEL=amazon.titan-embed-text-v2:0

# PDF Configuration
PDF_PATH=path/to/your/medical_device_document.pdf
```

## 🎯 Usage

### Running with Chainlit UI (Recommended)

```bash
chainlit run multi_agent_rag.py -w
```

This will:
1. Start the Chainlit web server
2. Open your browser automatically
3. Initialize the multi-agent system
4. Load and process your PDF document

The `-w` flag enables auto-reload on code changes.

### Access the Application

- **Default URL**: http://localhost:8000
- **Custom port**: Use `chainlit run multi_agent_rag.py --port 8080`

## 🎨 User Interface Features

### Chat Interface
- Clean, modern design
- Real-time message streaming
- Syntax highlighting for code
- Markdown support

### Agent Workflow Visualization
Each response shows:
- Which agent processed the query
- Retrieval statistics
- Step-by-step workflow

Example:
```
🔄 Agent Workflow:
- Router: Classified as 'analytical', routing to 'analytical_agent'
- Retrieval: Found 6 relevant sections
- Analytical Agent: Generated response
```

## 🔧 Architecture

### Query Flow

```
User Query
    ↓
Router Agent (Classifies query type)
    ↓
    ├─→ Greeting? → Conversational Agent → Response
    └─→ Document Query? → Retrieval Agent
                              ↓
                    ├─→ Factual Agent (Simple facts)
                    ├─→ Analytical Agent (Analysis/comparison)
                    └─→ Complex Agent (Multi-step reasoning)
                              ↓
                           Response
```

### Agent Specializations

1. **Router Agent**
   - Classifies queries using LLM
   - Routes to appropriate specialist
   - Handles greeting detection

2. **Retrieval Agent**
   - Vector similarity search
   - Adaptive chunk retrieval (4-8 chunks)
   - Context assembly

3. **Conversational Agent**
   - Handles greetings
   - Maintains friendly tone
   - No document retrieval needed

4. **Factual Agent**
   - Direct, concise answers
   - Strict context adherence
   - Citation-ready responses

5. **Analytical Agent**
   - Pattern identification
   - Comparative analysis
   - Information synthesis

6. **Complex Agent**
   - Step-by-step reasoning
   - Multi-part query handling
   - Detailed explanations

## 📊 Customization

### Adjusting Retrieval Parameters

In `PDFProcessor.__init__()`:

```python
PDFProcessor(
    pdf_path="your.pdf",
    chunk_size=1000,      # Adjust chunk size
    chunk_overlap=200     # Adjust overlap
)
```

### Modifying Agent Behavior

Each agent has its own system prompt in `create_multi_agent_nodes()`. Customize these to change behavior:

```python
system_prompt = """You are a factual information specialist...
[Your custom instructions here]
"""
```

### Changing LLM Parameters

In `MultiAgentPDFQA.__init__()`:

```python
self.llm = ChatOpenAI(
    model=COMPLETION_MODEL,
    temperature=0.3,      # Adjust for creativity (0.0-1.0)
    max_tokens=2000,      # Add token limit
    openai_api_key=self.api_key,
    openai_api_base=OPENAI_COMPAT_URL,
)
```

## 🐛 Troubleshooting

### Common Issues

1. **"PDF file not found"**
   ```bash
   # Verify PDF_PATH in .env
   export PDF_PATH=/absolute/path/to/document.pdf
   ```

2. **API Connection Errors**
   ```bash
   # Test your API endpoint
   curl -H "Authorization: Bearer $API_KEY" $OPENAI_COMPAT_URL/models
   ```

3. **Chainlit won't start**
   ```bash
   # Reinstall chainlit
   pip install --upgrade chainlit
   
   # Check port availability
   lsof -i :8000
   ```

4. **Memory Issues with Large PDFs**
   ```python
   # Reduce chunk size in PDFProcessor
   chunk_size=500  # Instead of 1000
   ```

## 📈 Performance Tips

1. **Cache Vector Store**: Save and load FAISS index to avoid reprocessing
   ```python
   # Save
   vector_store.save_local("faiss_index")
   
   # Load
   vector_store = FAISS.load_local("faiss_index", embeddings)
   ```

2. **Batch Processing**: Process multiple queries efficiently
   ```python
   for query in queries:
       response, steps = agent.ask(query)
   ```

3. **Adjust Retrieval**: Fewer chunks = faster responses
   ```python
   self.retriever = self.pdf_processor.get_retriever(k=3)  # Default is 4
   ```

## 🔐 Security Considerations

- Store API keys in `.env` file (never commit to git)
- Add `.env` to `.gitignore`
- Use environment-specific configurations
- Validate user inputs in production
- Implement rate limiting for public deployments

## 📝 Example Queries

### Factual Queries
- "What is the definition of risk management?"
- "List the key requirements for software validation"

### Analytical Queries
- "Compare the risk assessment approaches mentioned in the document"
- "What are the main differences between verification and validation?"

### Complex Queries
- "Explain the complete software lifecycle process and how risk management integrates at each stage"
- "What steps should I follow to implement a risk management system for medical device software?"

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add more specialized agents
- Implement agent memory/state persistence
- Add document comparison capabilities
- Create custom UI themes
- Add voice interaction support

## 📄 License

This project is provided as-is for educational and commercial use.

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [LangGraph](https://github.com/langchain-ai/langgraph)
- UI by [Chainlit](https://chainlit.io/)
- Vector storage by [FAISS](https://github.com/facebookresearch/faiss)

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Chainlit documentation: https://docs.chainlit.io
3. Check LangGraph documentation: https://langchain-ai.github.io/langgraph/

---

**Made with ❤️ for intelligent document understanding**

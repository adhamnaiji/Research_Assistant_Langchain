# 🔍 Intelligent Research Assistant

A production-ready Retrieval-Augmented Generation (RAG) system that enables natural language querying of PDF documents using Perplexity AI, LangChain, and FAISS vector search.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## ✨ Features

- 📄 **PDF Document Processing**: Upload and automatically parse PDF documents
- 🔎 **Semantic Search**: Vector-based similarity search using FAISS
- 🤖 **AI-Powered Answers**: Leverages Perplexity's Sonar models for intelligent responses
- 💬 **Interactive Chat Interface**: Streamlit-based conversational UI
- 📚 **Source Citations**: Transparent source attribution for all answers
- 🔧 **Extensible Tools**: Web search and calculator tools for enhanced capabilities
- 💾 **Persistent Storage**: Save and reload processed documents

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI (app.py)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
┌────────▼─────────┐          ┌─────────▼──────────┐
│ DocumentProcessor│          │  PerplexityRAG     │
│  (ETL Pipeline)  │          │  (Query Engine)    │
└────────┬─────────┘          └─────────┬──────────┘
         │                               │
    ┌────▼────┐                    ┌────▼────┐
    │  FAISS  │◄───────────────────┤Perplexity│
    │ Vector  │                    │   API    │
    │  Store  │                    └──────────┘
    └─────────┘
```

### Key Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **app.py** | Web interface & orchestration | Streamlit |
| **document_processor.py** | PDF parsing & vectorization | PyPDF, FAISS, HuggingFace |
| **llm_interface.py** | RAG implementation | LangChain, Perplexity API |
| **custom_tools.py** | Extensible agent tools | LangChain Tools |
| **test_system.py** | Integration testing | Python unittest |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Perplexity API key ([Get one here](https://www.perplexity.ai/))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/intelligent-research-assistant.git
cd intelligent-research-assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the project root:
```env
PERPLEXITY_API_KEY=your_api_key_here
```

### Running the Application

**Launch the Streamlit app:**
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload a Document

- Click **"Upload PDF Document"** in the sidebar
- Select your PDF file
- Click **"Process Document"** to begin processing

### 2. Ask Questions

Once processing is complete:
- Type your question in the chat input
- Press Enter to submit
- View the AI-generated answer with source citations

### 3. View Sources

- Click **"📚 View Sources"** to see the document excerpts used to generate the answer
- Each source includes page numbers and text snippets

### Example Queries

```
"What is the main topic of this document?"
"Summarize the key findings in section 3"
"What methodology was used in the research?"
"Compare the results from chapter 2 and chapter 4"
```

## 🔧 Configuration

### Model Selection

Choose between Perplexity models in `llm_interface.py`:

| Model | Context Window | Use Case | Cost |
|-------|---------------|----------|------|
| `sonar` | 8k tokens | Fast, general-purpose | $1/M tokens |
| `sonar-pro` | 32k tokens | Advanced reasoning | $3/M tokens |
| `sonar-reasoning` | 32k tokens | Deep analysis | $5/M tokens |

```python
rag = PerplexityRAG(vectorstore, model="sonar-pro")
```

### Chunking Parameters

Adjust in `document_processor.py` for different document types:

```python
processor = DocumentProcessor(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200     # Overlap between chunks (prevents context loss)
)
```

**Recommended Settings:**
- **Research Papers**: `chunk_size=1000`, `chunk_overlap=200`
- **Technical Manuals**: `chunk_size=1500`, `chunk_overlap=300`
- **Legal Documents**: `chunk_size=800`, `chunk_overlap=150`

### Retrieval Settings

Modify in `llm_interface.py`:

```python
self.retriever = self.vectorstore.as_retriever(
    search_type="similarity",  # Options: "similarity", "mmr", "similarity_score_threshold"
    search_kwargs={"k": 4}     # Number of chunks to retrieve (2-6 recommended)
)
```

## 🧪 Testing

Run the test suite to verify installation:

```bash
python test_system.py
```

**Prerequisites for testing:**
- Place a test PDF named `docker.pdf` in the project root
- Or modify `test_system.py` to use your own test file

**Expected Output:**
```
=== Testing Document Processing ===
✓ Vector store created successfully
✓ Vector store loaded successfully

=== Testing RAG System ===
Q: What is the main topic of this document?
A: [Generated answer]
Sources used: 4
Response time: 2.34s
```

## 📁 Project Structure

```
intelligent-research-assistant/
│
├── app.py                      # Streamlit web interface
├── document_processor.py       # PDF processing & vectorization
├── llm_interface.py            # RAG implementation
├── custom_tools.py             # Extensible agent tools
├── test_system.py              # Integration tests
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
└── README.md                   # This file
```

## 🔐 Security & Privacy

- **API Keys**: Stored in `.env` (never commit to version control)
- **Local Processing**: Document embeddings are generated locally
- **No Data Persistence**: Uploaded PDFs are temporarily stored and deleted after processing
- **Sandboxed Execution**: Custom tools use restricted `eval` for calculations

⚠️ **Important**: Always review documents before uploading. This system sends text chunks to Perplexity's API for processing.

## 🛠️ Advanced Features

### Adding Custom Tools

Extend functionality by adding tools in `custom_tools.py`:

```python
@tool
def fetch_stock_price(ticker: str) -> str:
    """Fetch real-time stock price for a given ticker symbol."""
    # Your implementation here
    return f"Stock price for {ticker}: $150.25"
```

### Saving Processed Documents

```python
# Save vectorstore for reuse
processor.save_vectorstore(vectorstore, path="my_vectorstore")

# Load previously processed vectorstore
vectorstore = processor.load_vectorstore(path="my_vectorstore")
```

### Multiple Document Processing

Combine multiple documents into one vectorstore:

```python
processor = DocumentProcessor()

# Process multiple PDFs
docs1 = processor.load_pdf("document1.pdf")
docs2 = processor.load_pdf("document2.pdf")

# Combine and vectorize
all_docs = docs1 + docs2
chunks = processor.chunk_documents(all_docs)
vectorstore = processor.create_vectorstore(chunks)
```

## 📊 Performance Optimization

### Embedding Model Options

Default model: `sentence-transformers/all-MiniLM-L6-v2` (80MB, fast)

For better accuracy (at the cost of speed):
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # 420MB, slower but more accurate
)
```

### GPU Acceleration

Enable GPU for faster embedding generation:

```bash
# Install GPU version of FAISS
pip uninstall faiss-cpu
pip install faiss-gpu
```

Update `document_processor.py`:
```python
model_kwargs={'device': 'cuda'}  # Changed from 'cpu'
```

## 🐛 Troubleshooting

### Common Issues



**Issue**: `PERPLEXITY_API_KEY not found in environment`
- Ensure `.env` file exists in project root
- Verify API key is correct
- Restart the application after adding the key

**Issue**: Slow processing speed
- Reduce `chunk_size` for faster processing
- Use GPU acceleration (see above)
- Consider using `faiss-gpu` instead of `faiss-cpu`






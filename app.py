import streamlit as st
from document_processor import DocumentProcessor
from llm_interface import PerplexityRAG
import os
import tempfile

# Page configuration
st.set_page_config(
    page_title="Intelligent Research Assistant",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
# Session state persists data across reruns
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processor' not in st.session_state:
    st.session_state.processor = DocumentProcessor()

def save_uploaded_file(uploaded_file):
    """
    Save uploaded file to temporary directory.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Path to saved file
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def process_document(file_path: str):
    """
    Process uploaded document and create RAG system.
    
    Args:
        file_path: Path to PDF file
    """
    with st.spinner("Processing document... This may take a minute."):
        try:
            # Process PDF and create vectorstore
            vectorstore = st.session_state.processor.process_pdf(file_path)
            st.session_state.vectorstore = vectorstore
            
            # Initialize RAG system
            st.session_state.rag_system = PerplexityRAG(vectorstore)
            
            st.success("✓ Document processed successfully!")
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

def display_sources(sources):
    """
    Display source documents used for answer.
    
    Args:
        sources: List of Document objects
    """
    with st.expander("📚 View Sources", expanded=False):
        for i, doc in enumerate(sources, 1):
            st.markdown(f"**Source {i}** (Page {doc.metadata.get('page', 'N/A')})")
            st.text(doc.page_content[:300] + "...")
            st.divider()

# App header
st.title("🔍 Intelligent Research Assistant")
st.markdown("""
### AI-Powered Document Analysis with RAG
Upload PDF documents and ask questions. The system uses:
- **Retrieval-Augmented Generation (RAG)** for accurate, grounded answers
- **Perplexity Sonar models** for intelligent responses
- **FAISS vector database** for efficient semantic search
""")

# Sidebar for document upload
with st.sidebar:
    st.header("📄 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Upload a PDF to analyze"
    )
    
    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                process_document(file_path)
                os.unlink(file_path)  # Clean up temp file
    
    st.divider()
    
    # Settings
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Select Model",
        ["sonar", "sonar-pro", "sonar-reasoning"],
        help="sonar: Fast, balanced | sonar-pro: Advanced capabilities | sonar-reasoning: Deep analysis"
    )
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
st.header("💬 Ask Questions")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            display_sources(message["sources"])

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    # Check if document is processed
    if st.session_state.rag_system is None:
        st.warning("⚠️ Please upload and process a document first!")
    else:
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_system.query(prompt)
                
                st.markdown(response["answer"])
                
                if response["sources"]:
                    display_sources(response["sources"])
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response["sources"]
                })

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
Built with LangChain, Perplexity AI, and Streamlit | 
<a href='https://github.com/yourusername/research-assistant'>View on GitHub</a>
</div>
""", unsafe_allow_html=True)

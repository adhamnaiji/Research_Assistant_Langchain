from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import os

class DocumentProcessor:
    """
    Handles document loading, chunking, and vectorization.
    
    This class implements the document processing pipeline:
    1. Load PDFs using PyPDFLoader
    2. Split documents into manageable chunks
    3. Generate embeddings using sentence transformers
    4. Store in FAISS vector database
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of text chunks (in characters)
            chunk_overlap: Overlap between consecutive chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings model
        # Using all-MiniLM-L6-v2: lightweight, fast, good performance
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize text splitter with smart chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Split by paragraphs first
        )
    
    def load_pdf(self, file_path: str) -> List:
        """
        Load PDF document and extract text.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects with page content and metadata
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            print(f"✓ Loaded {len(documents)} pages from {file_path}")
            return documents
        except Exception as e:
            print(f"✗ Error loading PDF: {str(e)}")
            raise
    
    def chunk_documents(self, documents: List) -> List:
        """
        Split documents into smaller chunks for better retrieval.
        
        Why chunking matters:
        - LLMs have limited context windows
        - Smaller chunks = more precise retrieval
        - Overlap preserves context across chunk boundaries
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks from documents")
        return chunks
    
    def create_vectorstore(self, chunks: List) -> FAISS:
        """
        Create FAISS vector database from document chunks.
        
        Process:
        1. Generate embeddings for each chunk
        2. Index embeddings in FAISS for fast similarity search
        3. Store original text in docstore
        
        Args:
            chunks: List of chunked Document objects
            
        Returns:
            FAISS vector store instance
        """
        try:
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            print(f"✓ Created vector store with {len(chunks)} documents")
            return vectorstore
        except Exception as e:
            print(f"✗ Error creating vector store: {str(e)}")
            raise
    
    def save_vectorstore(self, vectorstore: FAISS, path: str = "vectorstore"):
        """
        Save vector store to disk for persistence.
        
        Args:
            vectorstore: FAISS vector store instance
            path: Directory path to save the vector store
        """
        try:
            vectorstore.save_local(path)
            print(f"✓ Vector store saved to {path}")
        except Exception as e:
            print(f"✗ Error saving vector store: {str(e)}")
            raise
    
    def load_vectorstore(self, path: str = "vectorstore") -> FAISS:
        """
        Load existing vector store from disk.
        
        Args:
            path: Directory path where vector store is saved
            
        Returns:
            FAISS vector store instance
        """
        try:
            vectorstore = FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✓ Vector store loaded from {path}")
            return vectorstore
        except Exception as e:
            print(f"✗ Error loading vector store: {str(e)}")
            raise
    
    def process_pdf(self, file_path: str, save_path: str = "vectorstore") -> FAISS:
        """
        Complete pipeline: load PDF, chunk, vectorize, and save.
        
        Args:
            file_path: Path to PDF file
            save_path: Directory to save vector store
            
        Returns:
            FAISS vector store instance
        """
        print(f"\n📄 Processing document: {file_path}")
        documents = self.load_pdf(file_path)
        chunks = self.chunk_documents(documents)
        vectorstore = self.create_vectorstore(chunks)
        self.save_vectorstore(vectorstore, save_path)
        return vectorstore

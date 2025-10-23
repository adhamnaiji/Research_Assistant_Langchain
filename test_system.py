from document_processor import DocumentProcessor
from llm_interface import PerplexityRAG
import time

def test_document_processing():
    """Test document processing pipeline."""
    print("\n=== Testing Document Processing ===")
    
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    
    # Test with sample PDF
    sample_pdf = "docker.pdf"  # Replace with your test PDF
    vectorstore = processor.process_pdf(sample_pdf, save_path="test_vectorstore")
    
    print(f"✓ Vector store created successfully")
    
    # Test loading
    loaded_vectorstore = processor.load_vectorstore("test_vectorstore")
    print(f"✓ Vector store loaded successfully")
    
    return loaded_vectorstore

def test_rag_system(vectorstore):
    """Test RAG query system."""
    print("\n=== Testing RAG System ===")
    
    rag = PerplexityRAG(vectorstore, model="sonar")
    
    # Test queries
    test_questions = [
        "What is the main topic of this document?",
        "Summarize the key findings",
        "What methodology was used?"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        start_time = time.time()
        response = rag.query(question)
        elapsed_time = time.time() - start_time
        
        print(f"A: {response['answer'][:200]}...")
        print(f"Sources used: {response['num_sources']}")
        print(f"Response time: {elapsed_time:.2f}s")

if __name__ == "__main__":
    vectorstore = test_document_processing()
    test_rag_system(vectorstore)

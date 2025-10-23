from langchain_perplexity import ChatPerplexity
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class PerplexityRAG:
    """
    Integrates Perplexity LLM with RAG capabilities.
    
    This class combines:
    - Perplexity's Sonar models for generation
    - FAISS vector store for retrieval
    - Custom prompts for precise responses
    """
    
    def __init__(self, vectorstore: FAISS, model: str = "sonar"):
        """
        Initialize Perplexity RAG system.
        
        Args:
            vectorstore: FAISS vector store containing document embeddings
            model: Perplexity model to use ('sonar', 'sonar-pro', 'sonar-reasoning')
        """
        self.vectorstore = vectorstore
        
        # Initialize Perplexity LLM
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY not found in environment")
        
        self.llm = ChatPerplexity(
            model=model,
            temperature=0.2,  # Lower temperature for more focused responses
            pplx_api_key=api_key,
            max_tokens=1024
        )
        
        # Create retriever from vectorstore
        # k=4 means retrieve top 4 most relevant chunks
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Define custom prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Create RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff" = pass all retrieved docs to LLM
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )
    
    def _create_prompt_template(self) -> PromptTemplate:
        """
        Create custom prompt template for RAG.
        
        The prompt instructs the LLM to:
        1. Use retrieved context as primary source
        2. Acknowledge when information isn't in context
        3. Provide concise, accurate answers
        
        Returns:
            PromptTemplate instance
        """
        template = """You are an intelligent research assistant. Use the following pieces of context to answer the question at the end.

Context from documents:
{context}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, clearly state that
- Provide specific quotes or references when possible
- Be concise but comprehensive
- If you use external knowledge, clearly indicate it

Question: {question}

Detailed Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Process:
        1. Retriever finds relevant document chunks
        2. Chunks are passed to LLM as context
        3. LLM generates answer based on context
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing:
            - result: The generated answer
            - source_documents: Retrieved chunks used for answer
        """
        try:
            response = self.qa_chain.invoke({"query": question})
            return {
                "answer": response["result"],
                "sources": response["source_documents"],
                "num_sources": len(response["source_documents"])
            }
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "num_sources": 0
            }
    
    def get_relevant_chunks(self, question: str, k: int = 4) -> list:
        """
        Retrieve relevant document chunks without generating answer.
        
        Useful for:
        - Debugging retrieval quality
        - Showing users what context is being used
        - Adjusting retrieval parameters
        
        Args:
            question: Query to search for
            k: Number of chunks to retrieve
            
        Returns:
            List of Document objects with content and metadata
        """
        return self.retriever.get_relevant_documents(question)

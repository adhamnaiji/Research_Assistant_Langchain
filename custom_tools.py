from langchain.tools import Tool, tool
from langchain_perplexity import ChatPerplexity
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class CustomTools:
    """
    Custom tools to extend agent capabilities.
    
    Tools enable the agent to:
    - Search the web for current information
    - Perform calculations
    - Access external APIs
    """
    
    def __init__(self):
        """Initialize custom tools."""
        api_key = os.getenv("PERPLEXITY_API_KEY")
        self.search_llm = ChatPerplexity(
            model="sonar-pro",  # Pro model for better web search
            pplx_api_key=api_key,
            temperature=0.1
        )
    
    @tool
    def web_search(query: str) -> str:
        """
        Search the web for current information using Perplexity.
        
        Use this tool when:
        - User asks about recent events
        - Information isn't in uploaded documents
        - Current data is needed (stock prices, weather, news)
        
        Args:
            query: Search query
            
        Returns:
            Search results with citations
        """
        try:
            response = self.search_llm.invoke(
                f"Search the web and provide a comprehensive answer with citations: {query}"
            )
            return response.content
        except Exception as e:
            return f"Web search error: {str(e)}"
    
    @tool
    def calculate(expression: str) -> str:
        """
        Perform mathematical calculations.
        
        Use this tool for:
        - Arithmetic operations
        - Statistical calculations
        - Unit conversions
        
        Args:
            expression: Mathematical expression as string
            
        Returns:
            Calculation result
        """
        try:
            # Safe evaluation using Python's eval with restricted namespace
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    def get_tools(self) -> list:
        """
        Get list of all available tools.
        
        Returns:
            List of Tool objects
        """
        return [
            Tool(
                name="WebSearch",
                func=self.web_search,
                description="Search the web for current information. Input should be a search query string."
            ),
            Tool(
                name="Calculator",
                func=self.calculate,
                description="Perform mathematical calculations. Input should be a valid mathematical expression."
            )
        ]

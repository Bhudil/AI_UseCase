import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tavily import TavilyClient
from config.config import TAVILY_API_KEY


def search_web(query):
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=5,
            search_depth="basic",
            include_answer=True
        )
        
        return {
            "success": True,
            "answer": response.get("answer", ""),
            "results": response.get("results", [])
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def should_use_web_search(query):
    keywords = [
        "current", "latest", "recent", "today", "now", "news",
        "weather", "stock", "price", "score", "update",
        "2024", "2025", "this year", "this month", "this week"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in keywords)


def format_search_for_context(search_results):
    if not search_results.get("success"):
        return ""
    
    context_parts = []
    
    # Add answer
    if search_results.get("answer"):
        context_parts.append(f"Web Search Answer: {search_results['answer']}")
    
    # Add result snippets
    results = search_results.get("results", [])
    if results:
        context_parts.append("\nWeb Search Results:")
        for idx, result in enumerate(results[:3], 1):
            title = result.get("title", "")
            content = result.get("content", "")[:300]
            context_parts.append(f"{idx}. {title}: {content}")
    
    return "\n".join(context_parts)

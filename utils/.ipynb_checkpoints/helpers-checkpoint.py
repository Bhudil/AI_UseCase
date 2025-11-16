"""
Helper utilities for the chatbot application.
"""

import sys
import os
import hashlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import CHAT_HISTORY_CONTEXT_TURNS


def format_chat_history(history, max_turns=CHAT_HISTORY_CONTEXT_TURNS):
    """
    Format chat history for context injection.
    
    Args:
        history (list): List of message dictionaries
        max_turns (int): Maximum number of conversation turns to include
    
    Returns:
        str: Formatted chat history
    """
    if not history:
        return "No previous conversation."
    
    recent_history = history[-(max_turns * 2):]
    formatted = []
    
    for msg in recent_history:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            formatted.append(f"User: {content}")
        else:
            # Remove citations from history context
            clean_content = content.split("\n\n**Document Pages:**")[0]
            clean_content = clean_content.split("\n\n**Web Sources:**")[0]
            formatted.append(f"Assistant: {clean_content}")
    
    return "\n".join(formatted)


def get_cache_key(text):
    """
    Generate a cache key for a given text.
    
    Args:
        text (str): Text to hash
    
    Returns:
        str: SHA-256 hash of the text
    """
    return hashlib.sha256(text.lower().strip().encode()).hexdigest()


def format_sources(pages=None, web_results=None):
    """
    Format source citations for response.
    
    Args:
        pages (list): List of page numbers from documents
        web_results (dict): Web search results
    
    Returns:
        str: Formatted source citations
    """
    sources = []
    
    # Add document pages
    if pages:
        pages_str = ", ".join(map(str, pages))
        sources.append(f"\n\n**Document Pages:** {pages_str}")
    
    # Add web sources
    if web_results and web_results.get("success"):
        results = web_results.get("results", [])
        if results:
            sources.append("\n\n**Web Sources:**")
            for idx, result in enumerate(results[:3], 1):
                title = result.get("title", "")
                url = result.get("url", "")
                sources.append(f"{idx}. [{title}]({url})")
    
    return "\n".join(sources)
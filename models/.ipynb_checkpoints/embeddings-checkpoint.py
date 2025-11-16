"""
Embedding model initialization and management.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.embeddings import SentenceTransformerEmbeddings
from config.config import EMBEDDING_MODEL


def get_embedding_model():
    """
    Get or initialize the embedding model.
    
    Returns:
        SentenceTransformerEmbeddings: Initialized embedding model
    """
    try:
        embeddings = SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {str(e)}")
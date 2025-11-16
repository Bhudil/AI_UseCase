import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.embeddings import HuggingFaceEmbeddings
from config.config import EMBEDDING_MODEL


def get_embedding_model():
    """
    Get or initialize the embedding model.
    
    Returns:
        HuggingFaceEmbeddings: Initialized embedding model
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {str(e)}")

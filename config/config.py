import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API KEYS
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# MODEL SETTINGS
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_TEMPERATURE = 0.1

# RAG SETTINGS
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DB_FAISS_PATH = "vector_db/faiss_index"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
RETRIEVAL_K = 8

# RESPONSE MODE SETTINGS
RESPONSE_MODES = {
    "Concise": {
        "description": "Short, direct answers",
        "max_tokens": 512,
        "system_instruction": "Provide a brief, concise answer. Be direct and to the point. Use 2-3 sentences maximum."
    },
    "Detailed": {
        "description": "Comprehensive, in-depth responses",
        "max_tokens": 2048,
        "system_instruction": "Provide a comprehensive, detailed answer. Include explanations, examples, and context where relevant."
    }
}

# APPLICATION SETTINGS
PAGE_TITLE = "Document Q&A"
PAGE_ICON = "🔍"
LAYOUT = "wide"
MAX_CHAT_HISTORY = 20
CHAT_HISTORY_CONTEXT_TURNS = 3
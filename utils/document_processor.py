"""
Document processing utilities for RAG system.
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, DB_FAISS_PATH
from models.embeddings import get_embedding_model


def process_document(file):
    """
    Process uploaded document and create FAISS index.
    
    Args:
        file: Uploaded file object
    
    Returns:
        tuple: (vectorstore, bm25_index, corpus_docs, success_message)
    """
    try:
        embeddings = get_embedding_model()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        
        try:
            # Load PDF
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            docs = text_splitter.split_documents(pages)
            
            # Create BM25 index
            corpus_texts = [doc.page_content for doc in docs]
            bm25 = BM25Okapi([text.split() for text in corpus_texts])
            
            # Create FAISS vectorstore
            vectorstore = FAISS.from_documents(docs, embeddings)
            
            # Save to disk
            os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
            vectorstore.save_local(DB_FAISS_PATH)
            
            message = f"✅ Processed {len(pages)} pages into {len(docs)} chunks"
            
            return vectorstore, bm25, docs, message
        
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        raise Exception(f"Error processing document: {str(e)}")


def load_existing_vectorstore():
    """
    Load existing vectorstore from disk.
    
    Returns:
        FAISS vectorstore or None
    """
    try:
        if os.path.exists(DB_FAISS_PATH):
            embeddings = get_embedding_model()
            vectorstore = FAISS.load_local(
                DB_FAISS_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vectorstore
        return None
    except Exception as e:
        print(f"Error loading vectorstore: {str(e)}")
        return None
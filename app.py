import streamlit as st
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, MAX_CHAT_HISTORY,
    RESPONSE_MODES, GROQ_API_KEY, TAVILY_API_KEY
)
from models.llm import get_chatgroq_model, get_response_mode_instruction
from models.embeddings import get_embedding_model
from utils.document_processor import process_document, load_existing_vectorstore
from utils.retriever import retrieve_context
from utils.web_search import search_web, should_use_web_search, format_search_for_context
from utils.helpers import format_chat_history, get_cache_key, format_sources
from utils.question_generator import generate_insightful_questions, generate_document_summary

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize all session state variables"""
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    
    if "bm25_index" not in st.session_state:
        st.session_state.bm25_index = None
    
    if "corpus_docs" not in st.session_state:
        st.session_state.corpus_docs = []
    
    if "query_cache" not in st.session_state:
        st.session_state.query_cache = {}
    
    if "response_mode" not in st.session_state:
        st.session_state.response_mode = "Detailed"
    
    if "embeddings" not in st.session_state:
        try:
            st.session_state.embeddings = get_embedding_model()
        except Exception as e:
            st.error(f"Failed to load embedding model: {str(e)}")
    
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = []
    
    if "show_summary" not in st.session_state:
        st.session_state.show_summary = False
    
    # Try to load existing FAISS index
    if st.session_state.faiss_index is None:
        try:
            loaded_vectorstore = load_existing_vectorstore()
            if loaded_vectorstore:
                st.session_state.faiss_index = loaded_vectorstore
        except Exception:
            pass

def generate_response(query, response_mode):
    cache_key = get_cache_key(f"{query}_{response_mode}")
    if cache_key in st.session_state.query_cache:
        return st.session_state.query_cache[cache_key]
    
    try:
        history_context = format_chat_history(st.session_state.chat_history)
        instruction = get_response_mode_instruction(response_mode)
        rag_context = ""
        pages = []
        if st.session_state.faiss_index:
            rag_context, pages = retrieve_context(
                query,
                st.session_state.faiss_index,
                st.session_state.bm25_index,
                st.session_state.corpus_docs
            )

        web_results = None
        web_context = ""
        if should_use_web_search(query):
            with st.spinner("Searching the web..."):
                web_results = search_web(query)
                if web_results.get("success"):
                    web_context = format_search_for_context(web_results)

        prompt = f"""You are a helpful AI assistant. {instruction}

Previous conversation:
{history_context}

"""
        
        if rag_context:
            prompt += f"""Document Context:
{rag_context}

"""
        
        if web_context:
            prompt += f"""{web_context}

"""
        
        prompt += f"""User Question: {query}

Answer:"""

        llm = get_chatgroq_model(response_mode)
        response = llm.invoke(prompt).content
        
        sources = format_sources(pages=pages if rag_context else None, web_results=web_results)
        response += sources

        st.session_state.query_cache[cache_key] = response
        
        return response
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def render_sidebar():
    """Render sidebar with controls"""
    
    with st.sidebar:
        st.title("Settings")
        
        # Response Mode Selection
        st.subheader("Response Mode")
        response_mode = st.selectbox(
            "Choose response style:",
            options=list(RESPONSE_MODES.keys()),
            index=list(RESPONSE_MODES.keys()).index(st.session_state.response_mode),
            help="Select how detailed you want the responses to be"
        )
        
        # Update session state if changed
        if response_mode != st.session_state.response_mode:
            st.session_state.response_mode = response_mode
            st.session_state.query_cache = {}
        
        mode_desc = RESPONSE_MODES[response_mode]["description"]
        st.caption(mode_desc)
        
        st.divider()
        
        # Document Upload Section
        st.subheader("Knowledge Base")
        
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["pdf"],
            help="Upload PDF files to add to knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document", type="primary", use_container_width=True):
                with st.spinner("Processing document..."):
                    try:
                        vectorstore, bm25, docs, message = process_document(uploaded_file)
                        
                        st.session_state.faiss_index = vectorstore
                        st.session_state.bm25_index = bm25
                        st.session_state.corpus_docs = docs
                        st.session_state.query_cache = {}
                        
                        # Generate insightful questions
                        doc_content = "\n\n".join([doc.page_content for doc in docs[:20]])
                        st.session_state.suggested_questions = generate_insightful_questions(doc_content, 3)
                        
                        st.success(message)
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Show RAG status
        if st.session_state.faiss_index:
            st.success("Knowledge base loaded")
            st.caption(f"{len(st.session_state.corpus_docs)} chunks indexed")
            
            # Summarize button
            st.divider()
            if st.button("Generate Summary", use_container_width=True):
                with st.spinner("Generating summary..."):
                    doc_content = "\n\n".join([doc.page_content for doc in st.session_state.corpus_docs[:20]])
                    summary = generate_document_summary(doc_content)
                    st.session_state.show_summary = True
                    st.session_state.summary_content = summary
                    st.rerun()
        else:
            st.info("Upload a document to enable RAG")
        
        st.divider()
        
        # Chat Controls
        st.subheader("Chat Controls")
        
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.query_cache = {}
            st.rerun()
        
        if st.button("Reset Knowledge Base", use_container_width=True):
            st.session_state.faiss_index = None
            st.session_state.bm25_index = None
            st.session_state.corpus_docs = []
            st.session_state.query_cache = {}
            st.session_state.suggested_questions = []
            st.success("Knowledge base reset")
            st.rerun()


def render_chat_interface():
    """Render main chat interface"""
    
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    
    # Show summary if generated
    if st.session_state.get('show_summary', False):
        with st.expander("Document Summary", expanded=True):
            st.markdown(st.session_state.get('summary_content', ''))
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Close"):
                    st.session_state.show_summary = False
                    st.rerun()
            with col2:
                st.download_button(
                    label="Download Summary",
                    data=st.session_state.get('summary_content', ''),
                    file_name="document_summary.md",
                    mime="text/markdown"
                )
    
    # Display insightful questions after document upload
    if st.session_state.suggested_questions and st.session_state.faiss_index:
        st.subheader("Suggested Questions")
        st.caption("Click on any question to explore the document")
        
        for idx, question in enumerate(st.session_state.suggested_questions):
            if st.button(question, key=f"q_{idx}", use_container_width=True):
                # Add question to chat
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                # Generate response
                response = generate_response(question, st.session_state.response_mode)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                st.rerun()
        
        st.divider()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        
        # Validate query
        if not prompt.strip():
            st.error("Please enter a valid question")
            return
        
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt, st.session_state.response_mode)
                st.markdown(response)
        
        # Add assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Trim history if too long
        if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
            st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]


def render_instructions():
    
    st.title("Instructions")
    
    st.markdown("""
    ## Welcome to Research Assistant & Document Q&A
    
    This intelligent chatbot combines document analysis (RAG) with real-time web search 
    to provide accurate, contextual answers to your questions.
    
    ### Quick Start
    
    1. Set up API Keys - Add to `.env` file
    2. Upload documents to build your knowledge base
    3. Choose response mode (Concise or Detailed)
    4. Start asking questions
    
    ---
    
    ## API Key Setup
    
    Create a `.env` file in your project root:
```env
    GROQ_API_KEY=your_groq_key_here
    TAVILY_API_KEY=your_tavily_key_here
```
    
    **Get Your Keys:**
    - Groq: https://console.groq.com/keys
    - Tavily: https://tavily.com/
    
    ---
    
    ## Features
    
    ### Document Q&A (RAG)
    - Upload PDF files
    - Ask questions about document content
    - Get answers with page citations
    - Uses hybrid search (semantic + keyword)
    
    ### Web Search Integration
    - Automatically searches web for current information
    - Triggered by keywords like "latest", "current", "today"
    - Provides source links
    - Combines with document knowledge
    
    ### Response Modes
    - Concise: Quick, direct answers
    - Detailed: Comprehensive explanations with context
    
    ### Insightful Questions
    - Auto-generated questions after document upload
    - Click any question to explore the document
    - Questions are tailored to document content
    
    ### Document Summary
    - Generate comprehensive summaries
    - Download summaries in markdown format
    - Includes key points and conclusions
    
    ---
    
    ## Example Queries
    
    **With Documents:**
    - "What are the main findings in section 3?"
    - "Summarize the methodology"
    - "What does the contract say about payment terms?"
    
    **With Web Search:**
    - "What are the latest developments in AI?"
    - "Current weather in New York"
    - "Recent news about electric vehicles"
    
    **Combined:**
    - "Compare the document's approach with current industry standards"
    - "How does this research relate to recent findings?"
    """)

def main():

    initialize_session_state()

    if not GROQ_API_KEY or not TAVILY_API_KEY:
        st.error("Missing API keys! Please add GROQ_API_KEY and TAVILY_API_KEY to your .env file")
        st.info("See Instructions page for setup details")

    with st.sidebar:
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["Chat", "Instructions"],
            index=0
        )

    if page == "Chat":
        render_sidebar()

    if page == "Chat":
        render_chat_interface()
    else:
        render_instructions()


if __name__ == "__main__":
    main()
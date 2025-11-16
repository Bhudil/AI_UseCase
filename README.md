# Research Assistant & Document Q&A Chatbot

AI-powered document analysis system combining Retrieval-Augmented Generation (RAG) with real-time web search capabilities.

## Live Demo

**Deployed Application:** [[LINK](https://aiusecase-jctgzwaosfb7zyrhpbgbrl.streamlit.app/)]

## Features

- Document Q&A using hybrid retrieval (FAISS + BM25)
- Real-time web search integration via Tavily API
- Dual response modes (Concise/Detailed)
- Automatic question generation from documents
- Document summarization (On top of the page and downloadable)
- Source attribution with page numbers and URLs

## Prerequisites

- Python 3.9 or higher
- pip package manager
- Virtual environment

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Bhudil/AI_UseCase.git
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

**Obtain API Keys:**
- Groq: https://console.groq.com/keys
- Tavily: https://tavily.com

## Usage

### Run Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Basic Workflow

1. Upload a PDF document using the sidebar
2. Click "Process Document" to index the content
3. Use suggested questions or enter your own queries
4. Toggle between Concise and Detailed response modes
5. Click "Generate Summary" for document overview

## Project Structure

```
research-assistant-chatbot/
├── config/
│   └── config.py              # Configuration and API key management
├── models/
│   ├── __init__.py
│   ├── llm.py                 # LLM initialization (Groq)
│   └── embeddings.py          # Embedding model setup
├── utils/
│   ├── __init__.py
│   ├── document_processor.py  # PDF processing and chunking
│   ├── retriever.py           # Hybrid retrieval implementation
│   ├── web_search.py          # Tavily web search integration
│   ├── helpers.py             # Utility functions
│   └── question_generator.py  # Question and summary generation
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── .env                       # API keys (create this)
├── .gitignore
└── README.md
```

## Configuration

Edit `config/config.py` to modify:

- LLM model settings (temperature, max tokens)
- RAG parameters (chunk size, retrieval count)
- Response mode configurations
- Web search settings

## Technical Stack

- **Frontend:** Streamlit
- **LLM:** Groq (llama-3.1-8b-instant)
- **Embeddings:** HuggingFace Sentence Transformers
- **Vector Database:** FAISS
- **Keyword Search:** BM25
- **Web Search:** Tavily API
- **Document Processing:** PyPDF

## Deployment

### Streamlit Cloud

1. Push code to GitHub repository
2. Connect repository at https://streamlit.io/cloud
3. Configure secrets in Streamlit Cloud dashboard:
   ```
   GROQ_API_KEY = "your_key"
   TAVILY_API_KEY = "your_key"
   ```
4. Deploy application

### Local Deployment

```bash
streamlit run app.py --server.port 8501
```

## Dependencies

Core packages and versions specified in `requirements.txt`:
- streamlit==1.31.0
- langchain==0.1.20
- langchain-groq==0.1.3
- sentence-transformers==2.3.1
- faiss-cpu==1.7.4
- rank-bm25==0.2.2
- pypdf==3.17.0
- tavily-python==0.3.3

## Acknowledgments

Built using LangChain, Groq, Tavily, FAISS, and Streamlit frameworks.

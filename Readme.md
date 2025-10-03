# RAG Document Chatbot

A full-stack Retrieval-Augmented Generation (RAG) chatbot designed to provide intelligent answers to user queries based *only* on the content of uploaded documents. It uses a FastAPI backend for document processing and AI inference, and a modern frontend (likely React/Vite based on `vite.config.js`) for user interaction.

## ✨ Features

*   **Document Upload**: Upload PDF and TXT files.
*   **Intelligent Chunking**: Documents are split into optimized chunks for efficient retrieval.
*   **Hybrid Retrieval**: Combines semantic search (embeddings) and keyword search (BM25) for highly relevant context retrieval.
*   **Local Embedding Model**: Utilizes `sentence-transformers/all-MiniLM-L6-v2` for local, cost-effective, and private document embeddings.
*   **Google Gemini Pro LLM**: Leverages the powerful Google Gemini Pro (or Gemini Flash) model for answer generation.
*   **Chat History**: Maintains conversation context within a session.
*   **Contextual Answering**: LLM answers are strictly grounded in the retrieved document context.
*   **Clear Session Management**: Easily clear document and chat history for a session.

## 🚀 Getting Started

Follow these steps to set up and run the RAG Document Chatbot locally.

### Prerequisites

*   Python 3.9+ (Python 3.12.3 was used during development)
*   Node.js & npm (or yarn) for the frontend
*   Google Cloud Project & Gemini API Key (for LLM access)

### 1. Backend Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourGitHubUsername/RAG_Document_Chatbot.git
    cd RAG_Document_Chatbot
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv new_venv # Using 'new_venv' as per your project structure
    source new_venv/bin/activate # On Linux/macOS
    # new_venv\Scripts\activate # On Windows (Command Prompt)
    # new_venv\Scripts\Activate.ps1 # On Windows (PowerShell)
    ```

3.  **Install backend dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The `requirements.txt` should contain `sentence-transformers==2.2.2`, `torch>=2.2.0`, `huggingface_hub==0.17.3` among others, to ensure compatibility with the local embedding model.)*

4.  **Download NLTK data:**
    ```bash
    python -c "import nltk; nltk.download('punkt')"
    ```

5.  **Create a `.env` file:**
    In the root of your `RAG_Document_Chatbot` directory, create a file named `.env` and add your Google API key:
    ```
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
    GEMINI_LLM_MODEL="gemini-2.5-flash" # or "gemini-pro"
    GEMINI_EMBEDDING_MODEL="gemini-embedding-001" # This is used by config, but embedding.py uses local HF model
    # Adjust other settings as needed
    # MAX_FILE_SIZE_MB=100
    # UPLOAD_DIRECTORY="uploaded_documents"
    # CHAT_HISTORY_LIMIT=5
    ```
    **Important:** Replace `"YOUR_GEMINI_API_KEY"` with your actual Google Gemini API Key. Do **not** commit this file to Git!

6.  **Run the FastAPI backend:**
    ```bash
    uvicorn backend.app.main:app --reload
    ```
    The backend will typically run on `http://127.0.0.1:8000`.

### 2. Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Install frontend dependencies:**
    ```bash
    npm install
    # or
    yarn install
    ```

3.  **Run the frontend development server:**
    ```bash
    npm run dev
    # or
    yarn dev
    ```
    The frontend will typically run on `http://localhost:5173`.

## 🖥️ Usage

1.  Ensure both the backend (FastAPI) and frontend (Vite) servers are running.
2.  Open your web browser and navigate to `http://localhost:5173`.
3.  Upload a PDF or TXT document using the interface.
4.  Once the document is processed, you can start asking questions related to its content. The chatbot will retrieve relevant information and synthesize an answer.


## 🛠️ Technologies Used

### Backend
*   **Python 3.12+**
*   **FastAPI**: Web framework for building the API.
*   **LangChain**: Framework for developing applications powered by language models.
*   **`sentence-transformers`**: For local text embedding (using `all-MiniLM-L6-v2`).
*   **ChromaDB**: Lightweight, in-memory (or persistent) vector database for storing and retrieving document embeddings.
*   **Google Generative AI SDK (`google-generativeai`)**: For interacting with Gemini LLMs.
*   **`pypdf`**: For extracting text from PDF documents.
*   **`rank_bm25`**: For keyword-based retrieval (BM25 algorithm).
*   **`nltk`**: For sentence tokenization in chunking.
*   **`uvicorn`**: ASGI server for running FastAPI.

### Frontend
*   **React** (likely, given `.jsx` files)
*   **Vite**: Frontend build tool.
*   **npm/yarn**: Package manager.

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

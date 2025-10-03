import chromadb
from chromadb.utils import embedding_functions
from backend.app.core.config import settings
from backend.app.services.embedding import get_embedding_model_for_chroma # Import the new embedding function
from typing import Dict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# In-memory store for Chroma clients and collections per session
# For a production application, consider a persistent ChromaDB server or another vector DB
# Or, if using local persistent Chroma, ensure proper cleanup/loading.
SESSION_VECTOR_STORES: Dict[str, chromadb.Client] = {}
SESSION_CHAT_HISTORY: Dict[str, List[BaseMessage]] = {} # Store chat history as LangChain messages

def get_chroma_client_for_session(session_id: str) -> chromadb.Client:
    """
    Returns an in-memory ChromaDB client for a given session ID.
    If a client for the session doesn't exist, it creates one.
    """
    if session_id not in SESSION_VECTOR_STORES:
        # Use a new in-memory client for each session
        # For persistent storage, you'd initialize with a path: chromadb.PersistentClient(path="/path/to/db")
        SESSION_VECTOR_STORES[session_id] = chromadb.Client()
        print(f"Created new in-memory Chroma client for session: {session_id}")
    return SESSION_VECTOR_STORES[session_id]

def get_or_create_collection(client: chromadb.Client, collection_name: str) -> chromadb.Collection:
    """
    Gets an existing Chroma collection or creates a new one with the appropriate
    embedding function.
    """
    # Use the Google Generative AI embedding function
    google_ef = get_embedding_model_for_chroma()
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=google_ef # Pass the instantiated embedding function
    )

def add_message_to_history(session_id: str, message: BaseMessage):
    """Adds a message to the session's chat history, enforcing a limit."""
    if session_id not in SESSION_CHAT_HISTORY:
        SESSION_CHAT_HISTORY[session_id] = []
    
    SESSION_CHAT_HISTORY[session_id].append(message)
    
    # Enforce history limit
    if len(SESSION_CHAT_HISTORY[session_id]) > settings.CHAT_HISTORY_LIMIT:
        # Keep only the last N messages (N=limit)
        SESSION_CHAT_HISTORY[session_id] = SESSION_CHAT_HISTORY[session_id][-settings.CHAT_HISTORY_LIMIT:]
        
def get_session_history(session_id: str) -> List[BaseMessage]:
    """Retrieves the chat history for a given session."""
    return SESSION_CHAT_HISTORY.get(session_id, [])

def clear_session_history(session_id: str):
    """Clears the chat history for a given session."""
    if session_id in SESSION_CHAT_HISTORY:
        del SESSION_CHAT_HISTORY[session_id]
        print(f"Cleared chat history for session {session_id}")
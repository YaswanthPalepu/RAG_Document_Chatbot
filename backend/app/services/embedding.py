from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from backend.app.core.config import settings # Keep settings for potential future use or other configurations
from typing import List, Dict
import chromadb.utils.embedding_functions as embedding_functions
import time # Import time for delays
import os # For checking local model path

# Cache for embedding models to avoid reloading
_embedding_models: Dict[str, Embeddings] = {}

def get_embedding_model() -> Embeddings:
    """
    Loads and returns a local 'sentence-transformers/all-MiniLM-L6-v2' embedding model (HuggingFace wrapper).
    Caches the model to prevent redundant loading.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # HuggingFaceEmbeddings will automatically download if not found locally or in cache.

    if model_name not in _embedding_models:
        print(f"Loading embedding model: {model_name}...")
        # Ensure 'sentence-transformers' and 'torch' are installed for this to work.
        _embedding_models[model_name] = HuggingFaceEmbeddings(
            model_name=model_name,
            # If you want to use GPU, uncomment: model_kwargs={'device': 'cuda'}
            # If you want to explicitly set the cache directory: cache_folder="/path/to/your/cache"
            # encode_kwargs={'normalize_embeddings': False} # Set to True if normalization is desired, often good practice
        )
        print(f"Embedding model {model_name} loaded.")
    return _embedding_models[model_name]

def embed_documents(documents: List[str]) -> List[List[float]]:
    """
    Embeds a list of text documents into a list of vectors using the local 'all-MiniLM-L6-v2' model.
    Implements batching to manage memory and performance.
    """
    embedding_model = get_embedding_model()
    all_embeddings = []
    batch_size = 100
    delay_between_batches = 0.05 # Small delay, less critical for local models

    print(f"Embedding {len(documents)} documents in batches of {batch_size}...")

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        try:
            batch_embeddings = embedding_model.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            print(f"  Processed batch {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}")
            time.sleep(delay_between_batches) # Small wait to yield CPU/GPU
        except Exception as e:
            print(f"Error embedding batch {i // batch_size + 1}: {e}")
            raise

    return all_embeddings

def embed_query(query: str) -> List[float]:
    """
    Embeds a single query string into a vector using the local 'all-MiniLM-L6-v2' model.
    """
    embedding_model = get_embedding_model()
    return embedding_model.embed_query(query)

def get_embedding_model_for_chroma():
    """
    Returns an instance of ChromaDB's SentenceTransformerEmbeddingFunction
    which is compatible with ChromaDB client's `embedding_function` parameter,
    configured for 'sentence-transformers/all-MiniLM-L6-v2'.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Configuring ChromaDB for SentenceTransformer model: {model_name}")
    # ChromaDB's SentenceTransformerEmbeddingFunction will handle downloading/loading
    # if the model isn't already available in the environment.
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
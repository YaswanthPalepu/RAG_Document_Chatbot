# backend/app/services/retrieval.py

from typing import List
from chromadb import Collection # Keep for type hinting if desired, but don't call .get() on it
from langchain_core.documents import Document
from backend.app.core.db import get_chroma_client_for_session, get_or_create_collection
from backend.app.services.embedding import get_embedding_model, get_embedding_model_for_chroma
from backend.app.core.config import settings

# For hybrid retrieval
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import Chroma # Directly import Chroma vectorstore

def retrieve_relevant_chunks(session_id: str, query: str, k: int = 5) -> List[Document]:
    """
    Retrieves the most semantically relevant document chunks from the
    session-specific Chroma DB collection based on the user query,
    using a hybrid (semantic + keyword) retrieval approach.
    """
    client = get_chroma_client_for_session(session_id)
    collection_name = f"rag_collection_{session_id}"

    embedding_function_for_langchain = get_embedding_model()

    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function_for_langchain
    )

    # 1. Dense Retriever (Semantic Search using Embeddings)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 2. Sparse Retriever (Keyword Search using BM25)
    # Get all documents from Chroma to build the BM25 index.
    
    # Access the underlying chromadb.Collection object from the LangChain Chroma vectorstore
    chroma_collection_instance = vectorstore._collection

    # Now, call .get() on the *instance* of the collection to retrieve all IDs and data
    all_docs_in_session_raw = chroma_collection_instance.get(
        ids=chroma_collection_instance.get()['ids'], # Get all IDs from the collection instance
        include=['documents', 'metadatas']
    )
    
    bm25_docs = []
    if all_docs_in_session_raw and all_docs_in_session_raw['documents']:
        for i in range(len(all_docs_in_session_raw['ids'])):
            bm25_docs.append(Document(page_content=all_docs_in_session_raw['documents'][i],
                                      metadata=all_docs_in_session_raw['metadatas'][i]))

    if not bm25_docs:
        print("No documents found in session for BM25 retriever. Falling back to dense retrieval only.")
        return dense_retriever.invoke(query)

    sparse_retriever = BM25Retriever.from_documents(bm25_docs)
    sparse_retriever.k = k # Set k for sparse retriever as well

    # 3. Ensemble Retriever (Hybrid Search)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5] # Adjust weights as needed (0.5 for equal importance)
    )

    # Get results from the hybrid retriever
    retrieved_documents = ensemble_retriever.invoke(query)

    print(f"Retrieved {len(retrieved_documents)} documents using hybrid search for session {session_id}.")

    return retrieved_documents
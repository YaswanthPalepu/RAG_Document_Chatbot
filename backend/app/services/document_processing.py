import os
from typing import List
from langchain_core.documents import Document
from backend.app.core.db import get_chroma_client_for_session, get_or_create_collection
from backend.app.services.chunking import chunk_document
from backend.app.services.embedding import embed_documents
from backend.app.core.config import settings
import pypdf

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text content from a PDF file.
    """
    text = ""
    with open(file_path, 'rb') as f: # Open in binary read mode
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            if page.extract_text(): # Ensure text is not None
                text += page.extract_text() + "\n"
    return text

def process_and_index_document(session_id: str, file_path: str) -> bool:
    """
    Reads a document, chunks its content, embeds the chunks, and indexes them
    into a session-specific Chroma DB collection.
    Handles both PDF and TXT files.
    """
    try:
        # 1. Read the document content based on file type
        file_extension = os.path.splitext(file_path)[1].lower()
        text_content = ""

        if file_extension == '.pdf':
            print(f"Extracting text from PDF: {file_path}")
            text_content = extract_text_from_pdf(file_path)
        elif file_extension == '.txt':
            print(f"Reading text from TXT: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
        else:
            print(f"Unsupported file type for processing: {file_extension}")
            return False

        if not text_content.strip():
            print("Extracted or read text content is empty.")
            return False

        # 2. Chunk the document
        print(f"Chunking document for session {session_id}...")
        chunks: List[Document] = chunk_document(text_content)
        if not chunks:
            print("No chunks generated from the document.")
            return False

        # Prepare texts and metadatas for embedding and Chroma DB
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_metadatas = [chunk.metadata for chunk in chunks]
        # Add source information to metadata (e.g., original file name)
        original_filename = os.path.basename(file_path)
        for metadata in chunk_metadatas:
            metadata['source'] = original_filename # Store the original filename

        # 3. Embed the chunks
        print(f"Embedding {len(chunks)} chunks for session {session_id}...")
        chunk_vectors = embed_documents(chunk_texts)
        if not chunk_vectors:
            print("No embeddings generated for the chunks.")
            return False
        for i, (text, vector) in enumerate(zip(chunk_texts, chunk_vectors)):
            print(f"\nChunk {i + 1}:")
            print(f"Text: {text[:100]}...")  # Print first 100 characters of the chunk
            print(f"Vector (length {len(vector)}): {vector}")

        # 4. Store in Chroma DB
        print(f"The embeddings are converted into vectors: {len(chunk_vectors)}")
        client = get_chroma_client_for_session(session_id)
        collection_name = f"rag_collection_{session_id}"
        collection = get_or_create_collection(client, collection_name)

        print(f"Adding {len(chunks)} chunks to Chroma DB for session {session_id}...")
        # Chroma expects ids as strings, and the number of ids, documents, and embeddings must match.
        # Generate simple sequential IDs for the chunks
        chunk_ids = [f"chunk_{session_id}_{i}" for i in range(len(chunk_texts))] # Make IDs more unique per session

        collection.add(
            embeddings=chunk_vectors,
            documents=chunk_texts,
            metadatas=chunk_metadatas,
            ids=chunk_ids
        )
        print(f"Document indexed successfully for session {session_id}.")
        collections = client.list_collections()
        print("Collections in ChromaDB:")
        for col in collections:
            print(f"- {col.name}")

        return True

    except Exception as e:
        print(f"Error processing document for session {session_id}: {e}")
        return False
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary file: {file_path}")
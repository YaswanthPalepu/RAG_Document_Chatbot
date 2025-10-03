from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

def chunk_document(text_content: str) -> List[Document]:
    """
    Splits a large text document into smaller, more semantically coherent chunks.
    This version focuses on creating robust, self-contained chunks that are
    still small enough for efficient retrieval, without complex parent-child linking.
    """
    if not text_content.strip():
        return []

    # Use a single splitter for creating chunks suitable for both embedding and LLM context.
    # The chunk size should be large enough to contain sufficient context,
    # but small enough to fit multiple chunks into the LLM's context window
    # and to ensure relevant retrieval.
    # Adjust chunk_size based on your average document sentence length and model context window.
    # Gemini Pro has a context window of 32k tokens, so we can afford larger chunks
    # but smaller chunks help with more granular retrieval.
    # Sticking to a moderate size (e.g., 500-1000) for good balance.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Increased chunk size for Gemini Pro context
        chunk_overlap=50, # Maintained overlap to prevent splitting mid-sentence or mid-idea
        length_function=len,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""] # Prioritize natural breaks
    )

    # Create documents directly from the text content
    chunks: List[Document] = text_splitter.create_documents([text_content])

    # Optionally, add a simple ID to each chunk's metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"

    return chunks
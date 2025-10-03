import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

class Settings:
    """
    Configuration settings for the application.
    Loads values from environment variables or uses defaults.
    """
    # Google Gemini API Key
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")

    # Google Gemini Model settings
    GEMINI_LLM_MODEL: str = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")
    #GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    # File upload settings
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 100)) # Max 100 MB
    UPLOAD_DIRECTORY: str = "uploaded_documents" # Directory to temporarily store uploaded files

    # Chroma DB settings
    CHROMA_PERSIST_DIR: str = "chroma_db_data"

    # Chat history settings
    CHAT_HISTORY_LIMIT: int = int(os.getenv("CHAT_HISTORY_LIMIT", 5)) # Number of previous messages to remember

settings = Settings()

# Ensure the upload directory exists
os.makedirs(settings.UPLOAD_DIRECTORY, exist_ok=True)

# Check for Google API Key
if not settings.GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to your Gemini API key.")
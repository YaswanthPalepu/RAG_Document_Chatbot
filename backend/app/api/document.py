import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from backend.app.core.config import settings
from backend.app.services.document_processing import process_and_index_document
import uuid
from backend.app.core.db import clear_session_history # Import to clear history

router = APIRouter()

class UploadResponse(BaseModel):
    session_id: str
    message: str

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(None) # Can be provided by client or generated
):
    """
    Uploads a document (PDF/TXT), saves it, and processes it to create
    a vector index for a given session.
    """
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    if file.size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File size exceeds the limit of {settings.MAX_FILE_SIZE_MB} MB.")

    if not session_id:
        session_id = str(uuid.uuid4()) # Generate a new session ID if not provided

    # Ensure the upload directory exists
    os.makedirs(settings.UPLOAD_DIRECTORY, exist_ok=True)

    # Sanitize filename and create a secure path
    file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'txt'
    unique_filename = f"{session_id}.{file_extension}"
    file_location = os.path.join(settings.UPLOAD_DIRECTORY, unique_filename)

    # Save the uploaded file temporarily
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
        file.file.close()

    # Process and index the document asynchronously (or in a background task)
    indexing_success = process_and_index_document(session_id, file_location)

    if not indexing_success:
        raise HTTPException(status_code=500, detail="Failed to process and index the document.")

    # Clear chat history for this session if a new document is uploaded to it
    clear_session_history(session_id)

    return UploadResponse(session_id=session_id, message="Document indexed successfully! Ready to chat.")
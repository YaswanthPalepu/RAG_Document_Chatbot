from http.client import HTTPException
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api import document, chat
from backend.app.core.db import SESSION_VECTOR_STORES

app = FastAPI(
    title="RAG Chatbot API",
    description="Full-stack RAG chatbot backend using FastAPI and Hugging Face models.",
    version="1.0.0",
)

# Configure CORS
# In a production environment, restrict origins to your frontend's domain.
origins = [
    "http://localhost",
    "http://localhost:5173",  # Default Vite dev server port
    # Add other frontend origins if deployed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(document.router, prefix="/api/v1/document", tags=["Document"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])

# Add a session clear endpoint (as expected by your App.jsx)
@app.delete("/api/v1/document/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in SESSION_VECTOR_STORES:
        del SESSION_VECTOR_STORES[session_id]
        return {"message": f"Session {session_id} and its vector store cleared."}
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG Chatbot API! Visit /docs for API documentation."}
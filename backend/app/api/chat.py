import time
from typing import List
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel
from backend.app.services.retrieval import retrieve_relevant_chunks
from backend.app.services.generation import generate_answer_map_reduce
from backend.app.core.db import SESSION_VECTOR_STORES # To check if session exists
from backend.app.core.db import SESSION_VECTOR_STORES, get_session_history
from langchain_core.messages import HumanMessage, AIMessage
from backend.app.core.db import add_message_to_history # Import add_message_to_history

router = APIRouter()

class QueryRequest(BaseModel):
    session_id: str
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = [] # Added sources field, default to empty list

RATE_LIMIT_STORE = {}
RATE_LIMIT_DURATION = 60  # seconds
RATE_LIMIT_REQUESTS = 10  # requests per duration


@router.post("/ask", response_model=QueryResponse)
async def get_chat_answer(request: QueryRequest, http_request: Request):
    session_id = request.session_id
    user_query = request.query


     # --- 2. Basic Rate Limiting (In-memory, for demo) ---
    client_ip = http_request.client.host
    current_time = time.time()
    
    if client_ip not in RATE_LIMIT_STORE:
        RATE_LIMIT_STORE[client_ip] = []
    
    # Remove old timestamps outside the window
    RATE_LIMIT_STORE[client_ip] = [
        t for t in RATE_LIMIT_STORE[client_ip] if current_time - t < RATE_LIMIT_DURATION
    ]

    if len(RATE_LIMIT_STORE[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many requests. Please try again in {RATE_LIMIT_DURATION} seconds."
        )
    
    RATE_LIMIT_STORE[client_ip].append(current_time)
    # --- End Rate Limiting ---



    if session_id not in SESSION_VECTOR_STORES:
        raise HTTPException(status_code=404, detail="Session not found or document not indexed. Please upload a document.")

    # 1. Retrieve relevant chunks
    print(f"Retrieving chunks for session {session_id} with query: {user_query[:50]}...")
    relevant_docs = retrieve_relevant_chunks(session_id, user_query)

    # Extract source information for the frontend
    sources = list(set([doc.metadata.get('source', 'Unknown Source') for doc in relevant_docs if doc.metadata]))

    if not relevant_docs:
        return QueryResponse(answer="I couldn't find any relevant information in the document to answer that question. Please try rephrasing or ask a different question.")

    # 2. Get chat history for the session
    chat_history = get_session_history(session_id) # <--- ADD THIS LINE

    # 3. Generate answer using Map-Reduce QA (now correctly passing chat_history)
    print(f"Generating answer using {len(relevant_docs)} chunks for session {session_id}...")
    answer = generate_answer_map_reduce(user_query, relevant_docs, chat_history) # <--- UPDATE THIS LINE

    # 4. Add the user's question and the AI's answer to the session history
    # You'll need to import HumanMessage and AIMessage from langchain_core.messages
    
    add_message_to_history(session_id, HumanMessage(content=user_query))
    add_message_to_history(session_id, AIMessage(content=answer))

    return QueryResponse(answer=answer, sources=sources)
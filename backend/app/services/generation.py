from typing import List, Dict
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI # New: For Gemini
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from backend.app.core.config import settings

# Cache for LLM models to avoid reloading
_llm_models: Dict[str, ChatGoogleGenerativeAI] = {}

def get_llm_model() -> ChatGoogleGenerativeAI:
    """
    Loads and returns a Google Generative AI Chat model (e.g., Gemini Pro).
    Caches the model to prevent redundant loading.
    """
    model_name = settings.GEMINI_LLM_MODEL
    if model_name not in _llm_models:
        print(f"Loading Google Generative AI LLM model: {model_name}...")
        _llm_models[model_name] = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.2, # Lower temperature for more factual, less creative answers
            max_output_tokens=512 # Adjust as needed for verbosity
        )
        print(f"LLM model {model_name} loaded.")
    return _llm_models[model_name]

def generate_answer_stuff_chain(question: str, docs: List[Document], chat_history: List[BaseMessage]) -> str:
    """
    Generates an answer to a question by stuffing all relevant documents
    into the LLM's context, incorporating chat history and an improved prompt
    for synthesis and grounding.
    """
    if not docs:
        return "I don't have enough information from the document to answer that. Please upload a relevant document."

    llm = get_llm_model()

    context_for_llm = "\n\n---\n\n".join([doc.page_content for doc in docs])

    # Construct messages for the chat model, including history and RAG context
    messages = [
        SystemMessage(content="You are an AI assistant designed to answer questions based *only* on the provided document context. Read the context carefully and then provide a comprehensive and concise answer to the user's question. Synthesize information from the context as needed, but **do not make up facts or use external knowledge**. If the context does not contain enough information to answer the question, state that clearly and politely.")
    ]

    # Add previous chat history
    for msg in chat_history:
        messages.append(msg)

    # Add the current RAG context and user question
    messages.append(HumanMessage(content=f"Context:\n{context_for_llm}\n\nQuestion: {question}\n\nAnswer:"))

    try:
        # Invoke the chat model
        response = llm.invoke(messages)
        cleaned_answer = response.content.strip()

        # Improved check for insufficient information responses
        negative_phrases = [
            "i cannot answer", "not found in the document", "not enough information",
            "i could not find", "i am unable to provide", "i lack the necessary information",
            "i can't answer", "based on the provided context, i cannot"
        ]
        # Check if the response contains explicit negative phrases and is relatively short
        if any(phrase in cleaned_answer.lower() for phrase in negative_phrases) and len(cleaned_answer) < 200:
            return "I couldn't find a sufficient answer within the provided document or previous conversation history. Please try a different question or rephrase."

        return cleaned_answer

    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return "An error occurred while generating the answer. Please try again."


# Keep the old name for API compatibility
def generate_answer_map_reduce(question: str, docs: List[Document], chat_history: List[BaseMessage]) -> str:
    """Wrapper for compatibility, passes chat_history to the main function."""
    return generate_answer_stuff_chain(question, docs, chat_history)
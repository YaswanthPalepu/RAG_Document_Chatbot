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
            max_output_tokens=1024,
            top_p=0.8 # Adjust as needed for verbosity
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
        SystemMessage(content='''You are an AI assistant designed to answer questions based ONLY on the provided document context.
        Adopt a user-centric perspective, aiming to provide comprehensive, easy-to-understand answers in natural, flowing language.
        
        Guidelines for your response:
        1.  **Natural Language Focus:** Respond in plain, conversational English. Avoid using bullet points, asterisks (*), colons (:), or parentheses () to introduce or structure information unless absolutely necessary for clarity (e.g., a simple parenthetical explanation, but not for list items). Aim for full sentences and cohesive paragraphs.
        2.  **Completeness & Clarity:** Provide a full and coherent answer. Do not truncate your thoughts or sentences. Ensure the answer is clear, well-structured, and easy for the user to follow.
        3.  **Summarize & Synthesize:** Read the context carefully. Synthesize information from various parts of the context to form a complete picture, summarizing relevant points concisely.
        4.  **Grounding:** Do not make up facts or use external knowledge. Every piece of information in your answer must be traceable to the provided context. If the context does not contain enough information, state that clearly and politely.
        5.  **User's Perspective:** Anticipate what the user *needs* to know and phrase the answer in a way that directly addresses their implicit or explicit query.
        6.  **Code Examples (Conditional):** If the user's question explicitly asks for code, or if the context clearly provides code snippets that are essential for the answer, include them. Format code clearly using markdown code blocks (```language\ncode\n```). Otherwise, describe functionalities without showing code.
        7.  **Conciseness & Detail:** Be as concise as possible while providing sufficient detail to fully answer the question from the user's perspective. Avoid unnecessary verbosity.''')
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
        # negative_phrases = [
        #     "i cannot answer", "not found in the document", "not enough information",
        #     "i could not find", "i am unable to provide", "i lack the necessary information",
        #     "i can't answer", "based on the provided context, i cannot"
        # ]
        # # Check if the response contains explicit negative phrases and is relatively short
        # if any(phrase in cleaned_answer.lower() for phrase in negative_phrases) and len(cleaned_answer) < 200:
        #     return "I couldn't find a sufficient answer within the provided document or previous conversation history. Please try a different question or rephrase."

        return cleaned_answer

    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return "An error occurred while generating the answer. Please try again."


# Keep the old name for API compatibility
def generate_answer_map_reduce(question: str, docs: List[Document], chat_history: List[BaseMessage]) -> str:
    """Wrapper for compatibility, passes chat_history to the main function."""
    return generate_answer_stuff_chain(question, docs, chat_history)
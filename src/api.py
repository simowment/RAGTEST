#!/usr/bin/env python3
"""
FastAPI application to expose the RAG assistant as an API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from openai import RateLimitError
from .assistant import (
    vectorbt_mode,
    review_mode,
)
from .llm_manager import LLMManager, managed_chat_request
from contextlib import asynccontextmanager

# In-memory store for chat engines and the LLM manager
STATE = {
    "vectorbt_index": None,
    "llm_manager": None,
    "review_sessions": {}  # <--- nouveau
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("Initializing LLM Manager...")
    try:
        STATE["llm_manager"] = LLMManager()
    except ValueError as e:
        print(f"Error initializing LLM Manager: {e}")
        # This is a critical error, so we might want to stop the app from starting.
        # For now, we print and let it continue, but it will fail on the first request.
        # In a production setup, you might `raise` here to stop the server.
    
    yield
    # Code to run on shutdown (if any)
    print("Shutting down...")

# Initialize FastAPI app with the lifespan manager
app = FastAPI(
    title="VectorBT & Code Review RAG Assistant",
    description="An API to interact with the VectorBT documentation and review code snippets.",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class Query(BaseModel):
    question: str

class CodeReview(BaseModel):
    code: str
    question: str
    session_id: str = None  # <--- nouveau
from typing import List, Optional, Literal

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-xxx"
    object: str = "chat.completion"
    choices: List[dict]
    usage: Optional[dict] = None


@app.get("/", response_class=FileResponse)
async def read_index():
    """
    Serve the main web interface.
    """
    return FileResponse('static/index.html')

def get_index():
    """
    Manages the creation and retrieval of the VectorStoreIndex.
    """
    if STATE["vectorbt_index"] is None:
        print("Creating new VectorBT index...")
        STATE["vectorbt_index"] = vectorbt_mode(api_mode=True)
    return STATE["vectorbt_index"]

@app.post("/vectorbt/query", summary="Query the VectorBT documentation")
async def query_vectorbt(query: Query):
    """
    Ask a question about the VectorBT documentation and codebase.
    """
    if not STATE["llm_manager"]:
        raise HTTPException(status_code=500, detail="LLM Manager is not initialized. Check server logs.")

    index = get_index()
    try:
        return await managed_chat_request(index, query.question, STATE["llm_manager"])
    except RateLimitError as e:
        # This is the final error after all retries have failed
        raise HTTPException(status_code=429, detail=f"All API configurations are rate-limited. Last error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/review/code")
async def review_code(review: CodeReview):
    """
    Provide a code snippet and a question to get a review.
    """
    if not STATE["llm_manager"]:
        raise HTTPException(status_code=500, detail="LLM Manager is not initialized. Check server logs.")

    # Utilisez session_id pour retrouver ou créer l'historique
    session_id = review.session_id or "default"
    if session_id not in STATE["review_sessions"]:
        engine = review_mode(api_mode=True, code_snippet=review.code)
        STATE["review_sessions"][session_id] = engine
    else:
        engine = STATE["review_sessions"][session_id]

    try:
        return await managed_chat_request(engine, review.question, STATE["llm_manager"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


from uuid import uuid4
from fastapi import Request

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def openai_compatible_chat(req: ChatCompletionRequest, request: Request):
    """
    OpenAI-compatible chat endpoint for integration with Continue or other clients.
    """
    if not STATE["llm_manager"]:
        raise HTTPException(status_code=500, detail="LLM Manager is not initialized.")

    # Extraire tous les messages user pour les concaténer
    user_query = "\n".join(msg.content for msg in req.messages if msg.role == "user").strip()

    index = get_index()

    try:
        response = await managed_chat_request(index, user_query, STATE["llm_manager"])
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid4()}",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.response
                    },
                    "finish_reason": "stop"
                }
            ],
            usage=None  # Optionnel: tu peux logguer les tokens ici si dispo
        )
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"Rate limit: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
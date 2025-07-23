#!/usr/bin/env python3
"""
FastAPI application to expose the RAG assistant as an API.
"""

import os
import sys
from pathlib import Path

# Add the 'scripts' directory to the Python path
# This is necessary for the imports to work correctly when running the API from the root
sys.path.append(str(Path(__file__).parent / "scripts"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from openai import RateLimitError
from assistant import (
    vectorbt_mode,
    review_mode,
)
from llm_manager import LLMManager, managed_chat_request
from contextlib import asynccontextmanager

# In-memory store for chat engines and the LLM manager
STATE = {
    "vectorbt_chat_engine": None,
    "llm_manager": None
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

class Query(BaseModel):
    question: str

class CodeReview(BaseModel):
    code: str
    question: str

def get_chat_engine(mode, code_snippet=None):
    """
    Manages the creation and retrieval of chat engines.
    For VectorBT, it's a singleton. For code review, it's created on-demand.
    The LLM instance within the engine will be managed by the LLMManager.
    """
    if mode == "vectorbt":
        if STATE["vectorbt_chat_engine"] is None:
            print("Creating new VectorBT chat engine...")
            # The LLM is set globally in vectorbt_mode, but we will override it per-request
            STATE["vectorbt_chat_engine"] = vectorbt_mode(api_mode=True)
        return STATE["vectorbt_chat_engine"]
    elif mode == "review":
        print("Creating new on-demand code review engine...")
        # The LLM is set globally in review_mode, but we will override it per-request
        return review_mode(api_mode=True, code_snippet=code_snippet)

@app.post("/vectorbt/query", summary="Query the VectorBT documentation")
async def query_vectorbt(query: Query):
    """
    Ask a question about the VectorBT documentation and codebase.
    """
    if not STATE["llm_manager"]:
        raise HTTPException(status_code=500, detail="LLM Manager is not initialized. Check server logs.")

    engine = get_chat_engine("vectorbt")
    try:
        return await managed_chat_request(engine, query.question, STATE["llm_manager"])
    except RateLimitError as e:
        # This is the final error after all retries have failed
        raise HTTPException(status_code=429, detail=f"All API configurations are rate-limited. Last error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/review/code", summary="Review a code snippet")
async def review_code(review: CodeReview):
    """
    Provide a code snippet and a question to get a review.
    """
    if not STATE["llm_manager"]:
        raise HTTPException(status_code=500, detail="LLM Manager is not initialized. Check server logs.")

    engine = get_chat_engine("review", code_snippet=review.code)
    try:
        return await managed_chat_request(engine, review.question, STATE["llm_manager"])
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"All API configurations are rate-limited. Please try again later. Last error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 
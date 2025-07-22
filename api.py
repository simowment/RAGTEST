#!/usr/bin/env python3
"""
FastAPI application to expose the RAG assistant as an API.
"""

import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
from openai import RateLimitError
from scripts.assistant import (
    vectorbt_mode,
    review_mode,
)
from contextlib import asynccontextmanager

# In-memory store for chat engines and key management
# In a real-world scenario, you'd use a more persistent store like Redis
STATE = {
    "vectorbt_chat_engine": None,
    "api_keys": [],
    "current_key_index": 0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("Loading API keys...")
    key1 = os.getenv("OPENROUTER_API_KEY_1")
    key2 = os.getenv("OPENROUTER_API_KEY_2")
    
    if not key1:
        raise RuntimeError("OPENROUTER_API_KEY_1 environment variable is not set.")
        
    STATE["api_keys"] = [key.strip() for key in [key1, key2] if key and key.strip()]
    
    if not STATE["api_keys"]:
         raise RuntimeError("No OpenRouter API keys found in environment variables.")
         
    print(f"Loaded {len(STATE['api_keys'])} API key(s).")
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
    """
    if mode == "vectorbt":
        if STATE["vectorbt_chat_engine"] is None:
            print("Creating new VectorBT chat engine...")
            STATE["vectorbt_chat_engine"] = vectorbt_mode(api_mode=True)
        return STATE["vectorbt_chat_engine"]
    elif mode == "review":
        print("Creating new on-demand code review engine...")
        return review_mode(api_mode=True, code_snippet=code_snippet)

def switch_api_key(engine):
    """Switches the API key for the given chat engine."""
    keys = STATE["api_keys"]
    if len(keys) <= 1:
        # No other keys to switch to
        return False

    current_index = STATE["current_key_index"]
    new_index = (current_index + 1) % len(keys)
    new_key = keys[new_index]
    
    print(f"Rate limit hit. Switching from key index {current_index} to {new_index}.")

    # Re-create the LLM with the new key but same settings
    original_llm_settings = {
        "model": engine._llm.model,
        "temperature": engine._llm.temperature,
        "max_tokens": engine._llm.max_tokens,
        "context_window": engine._llm.context_window,
    }
    engine._llm = vectorbt_mode(api_mode=True)._llm.__class__(api_key=new_key, **original_llm_settings)
    STATE["current_key_index"] = new_index
    return True

async def handle_chat_request(engine, question):
    """
    A unified function to handle chat requests with fallback logic.
    """
    for _ in range(len(STATE["api_keys"])):
        try:
            response = await engine.achat(question)
            return {"response": str(response)}
        except RateLimitError:
            print("Caught RateLimitError in API request.")
            if not switch_api_key(engine):
                raise HTTPException(status_code=429, detail="All API keys are rate-limited. Please try again later.")
            # After switching, the loop will retry with the new key.
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # This should only be reached if all keys fail
    raise HTTPException(status_code=429, detail="All API keys are rate-limited after retries.")

@app.post("/vectorbt/query", summary="Query the VectorBT documentation")
async def query_vectorbt(query: Query):
    """
    Ask a question about the VectorBT documentation and codebase.
    """
    engine = get_chat_engine("vectorbt")
    return await handle_chat_request(engine, query.question)

@app.post("/review/code", summary="Review a code snippet")
async def review_code(review: CodeReview):
    """
    Provide a code snippet and a question to get a review.
    """
    engine = get_chat_engine("review", code_snippet=review.code)
    return await handle_chat_request(engine, review.question)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 
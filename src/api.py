#!/usr/bin/env python3
"""
FastAPI application to expose the RAG assistant as an API.
"""

from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from openai import RateLimitError
from typing import List, Optional
import base64
import io
import os
from PIL import Image
from llama_index.postprocessor.cohere_rerank import CohereRerank
from .assistant import (
    vectorbt_mode,
    review_mode,
    load_knowledge_base,
)
from .llm_manager import LLMManager, managed_chat_request, EnhancedChatWrapper
from .knowledge_bases import KnowledgeBaseManager, KnowledgeBaseType
from contextlib import asynccontextmanager

# In-memory store for chat engines and the LLM manager
STATE = {
    "knowledge_bases": {},  # Cache for loaded knowledge bases
    "llm_manager": None,
    "review_sessions": {},
    "kb_manager": None,
    "chat_histories": {}  # Historique de conversation par knowledge base
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("Initializing LLM Manager...")
    try:
        STATE["llm_manager"] = LLMManager()
        STATE["kb_manager"] = KnowledgeBaseManager()
        print(f"Available knowledge bases: {[kb.name for kb in STATE['kb_manager'].get_available_knowledge_bases()]}")
    except ValueError as e:
        print(f"Error initializing managers: {e}")
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

@app.post("/clear-history/{kb_id}")
async def clear_chat_history(kb_id: str):
    """Clear conversation history for a specific knowledge base."""
    if kb_id in STATE["chat_histories"]:
        STATE["chat_histories"][kb_id] = []
        return {"message": f"Chat history cleared for {kb_id}"}
    return {"message": "No history found"}

@app.get("/knowledge-bases")
async def get_knowledge_bases():
    """
    Get all available knowledge bases.
    Auto-builds missing indices.
    """
    if not STATE["kb_manager"]:
        raise HTTPException(status_code=500, detail="Knowledge base manager not initialized")
    
    kb_list = []
    for kb in STATE["kb_manager"].get_available_knowledge_bases():
        # Auto-build if missing (except code review)
        available = STATE["kb_manager"].auto_build_knowledge_base(kb.id)
        
        kb_list.append({
            "id": kb.id,
            "name": kb.name,
            "description": kb.description,
            "type": kb.type.value,
            "supports_images": kb.supports_images,
            "icon": kb.icon,
            "available": available
        })
    
    return {"knowledge_bases": kb_list}

def get_knowledge_base_index(kb_id: str):
    """
    Manages the creation and retrieval of knowledge base indices.
    Auto-builds missing indices.
    """
    if kb_id not in STATE["knowledge_bases"]:
        print(f"Loading knowledge base: {kb_id}")
        kb_config = STATE["kb_manager"].get_knowledge_base(kb_id)
        if not kb_config:
            raise ValueError(f"Knowledge base '{kb_id}' not found")
        
        if kb_config.type == KnowledgeBaseType.CODE_REVIEW:
            # Code review doesn't use persistent indices
            return None
        
        # Load the knowledge base (which now includes auto-build logic)
        try:
            if kb_id == "unified_strategy":
                from .assistant import load_unified_strategy_assistant
                index = load_unified_strategy_assistant(api_mode=True, llm_manager=STATE["llm_manager"])
            else:
                from .assistant import load_knowledge_base
                index = load_knowledge_base(kb_id, api_mode=True, llm_manager=STATE["llm_manager"])
            
            STATE["knowledge_bases"][kb_id] = index
        except Exception as e:
            print(f"Failed to load knowledge base '{kb_id}': {e}")
            raise ValueError(f"Failed to load knowledge base '{kb_id}': {e}")
    
    return STATE["knowledge_bases"][kb_id]

def process_images(images: List[UploadFile]) -> List[str]:
    """
    Process uploaded images and return base64 encoded strings.
    """
    processed_images = []
    for image in images:
        try:
            # Read image data
            image_data = image.file.read()
            
            # Open with PIL to validate and potentially resize
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Resize if too large (max 1024x1024)
            max_size = 1024
            if pil_image.width > max_size or pil_image.height > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert back to bytes
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=85)
            img_buffer.seek(0)
            
            # Encode to base64
            base64_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            processed_images.append(f"data:image/jpeg;base64,{base64_image}")
            
        except Exception as e:
            print(f"Error processing image {image.filename}: {e}")
            continue
    
    return processed_images

@app.post("/query/{kb_id}", summary="Query a specific knowledge base")
async def query_knowledge_base(
    kb_id: str,
    question: str = Form(""),
    images: List[UploadFile] = File(default=[])
):
    """
    Ask a question about a specific knowledge base, optionally with images.
    """
    if not STATE["llm_manager"] or not STATE["kb_manager"]:
        raise HTTPException(status_code=500, detail="Managers not initialized. Check server logs.")

    # Validate knowledge base
    kb_config = STATE["kb_manager"].get_knowledge_base(kb_id)
    if not kb_config:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_id}' not found")
    
    if not STATE["kb_manager"].knowledge_base_exists(kb_id):
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_id}' is not available")

    # Process images if any (only for multimodal-capable knowledge bases)
    processed_images = []
    if images and kb_config.supports_images:
        processed_images = process_images([img for img in images if img.filename])

    # Build the query with images
    full_query = question
    if processed_images:
        image_context = f"\n\nImages provided: {len(processed_images)} image(s) for analysis"
        full_query += image_context

    try:
        print(f"🔍 [DEBUG] Querying knowledge base: {kb_id}")
        print(f"🔍 [DEBUG] Question length: {len(question)}")
        print(f"🔍 [DEBUG] Images provided: {len(processed_images)}")
        
        index = get_knowledge_base_index(kb_id)
        
        # Récupérer ou créer l'historique pour cette knowledge base
        if kb_id not in STATE["chat_histories"]:
            STATE["chat_histories"][kb_id] = []
            print(f"🔍 [DEBUG] Created new chat history for {kb_id}")
        
        # Créer un wrapper avec historique si c'est un index
        if hasattr(index, 'as_chat_engine'):
            print(f"🔍 [DEBUG] Using standard index with EnhancedChatWrapper")
            llm = STATE["llm_manager"].get_llm()
            base_chat_engine = index.as_chat_engine(
                chat_mode="context",
                similarity_top_k=15,
                node_postprocessors=[
                    CohereRerank(api_key=os.getenv("COHERE_API_KEY"), top_n=5)
                ],
                llm=llm
            )
            enhanced_engine = EnhancedChatWrapper(base_chat_engine, STATE["chat_histories"][kb_id])
            response = await enhanced_engine.achat(question)
            response_dict = {"response": response.response}
        else:
            print(f"🔍 [DEBUG] Using custom chat engine (CodeReviewChat or UnifiedStrategyChat)")
            response_dict = await managed_chat_request(index, full_query, STATE["llm_manager"])
        
        # Add metadata to response
        response_dict["knowledge_base"] = kb_config.name
        response_dict["supports_images"] = kb_config.supports_images
        if processed_images:
            response_dict["images_processed"] = len(processed_images)
            
        return response_dict
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"All API configurations are rate-limited. Last error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# Keep the old endpoint for backward compatibility
@app.post("/vectorbt/query", summary="Query the VectorBT documentation (deprecated)")
async def query_vectorbt(
    question: str = Form(""),
    images: List[UploadFile] = File(default=[])
):
    """
    Ask a question about the VectorBT documentation and codebase, optionally with images.
    This endpoint is deprecated. Use /query/vectorbt instead.
    """
    return await query_knowledge_base("vectorbt", question, images)

@app.post("/review/code")
async def review_code(
    code: str = Form(...),
    question: str = Form(""),
    session_id: str = Form(None),
    images: List[UploadFile] = File(default=[])
):
    """
    Provide a code snippet and a question to get a review, optionally with images.
    """
    if not STATE["llm_manager"]:
        raise HTTPException(status_code=500, detail="LLM Manager is not initialized. Check server logs.")

    # Process images if any
    processed_images = []
    if images:
        processed_images = process_images([img for img in images if img.filename])

    # Build the query with images
    full_question = question
    if processed_images:
        image_context = f"\n\nImages provided: {len(processed_images)} image(s) for additional context"
        full_question += image_context

    # Utilisez session_id pour retrouver ou créer l'historique
    session_id = session_id or "default"
    if session_id not in STATE["review_sessions"]:
        engine = review_mode(api_mode=True, code_snippet=code)
        STATE["review_sessions"][session_id] = engine
    else:
        engine = STATE["review_sessions"][session_id]

    try:
        response = await managed_chat_request(engine, full_question, STATE["llm_manager"])
        
        # Add image information to response if images were provided (response is a dict)
        if processed_images:
            response["images_processed"] = len(processed_images)
            
        return response
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
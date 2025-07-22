#!/usr/bin/env python3
"""
Unified RAG assistant for Vectorbt documentation and on-the-fly code review.
"""

import os
import sys
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from openai import RateLimitError
import chromadb

# Load environment variables
load_dotenv()

# Constants
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "tngtech/deepseek-r1t2-chimera:free")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")

def setup_global_settings(context_window=163840, max_tokens=4096):
    """
    Set up the global settings for LlamaIndex, adaptable for different modes.
    """
    primary_key = os.getenv("OPENROUTER_API_KEY_1")
    if not primary_key:
        print("Error: OPENROUTER_API_KEY_1 environment variable not set.", file=sys.stderr)
        sys.exit(1)
        
    if not os.getenv("COHERE_API_KEY"):
        print("Error: COHERE_API_KEY environment variable not set.", file=sys.stderr)
        print("Please get your key from https://dashboard.cohere.com/api-keys", file=sys.stderr)
        sys.exit(1)

    Settings.llm = OpenRouter(
        api_key=primary_key,
        model=OPENROUTER_MODEL, 
        temperature=0.1, 
        max_tokens=max_tokens,
        context_window=context_window
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

def get_multiline_input(prompt):
    """
    Prompts the user to paste multiline text and reads it until they signal the end.
    """
    print(prompt)
    print("When you are finished, press Ctrl+Z followed by Enter (Windows) or Ctrl+D (Mac/Linux).")
    lines = []
    while True:
        try:
            line = input()
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)

def run_chat_loop(chat_engine, session_name):
    """
    Main chat loop for a given chat engine.
    """
    # Get keys for fallback mechanism
    key1 = os.getenv("OPENROUTER_API_KEY_1")
    key2 = os.getenv("OPENROUTER_API_KEY_2")
    api_keys = [key.strip() for key in [key1, key2] if key and key.strip()]
    
    if not api_keys:
        print("Warning: No OpenRouter API keys found. Fallback is disabled.", file=sys.stderr)

    current_key_index = 0
    
    # Store original LLM settings to re-create it with a new key
    original_llm_settings = {
        "model": Settings.llm.model,
        "temperature": Settings.llm.temperature,
        "max_tokens": Settings.llm.max_tokens,
        "context_window": Settings.llm.context_window,
    }

    while True:
        question = input(f"\nYour question for {session_name} (or 'exit' to quit): ")
        if question.strip().lower() in ['exit', 'quit', 'q']:
            break
        if not question.strip():
            continue

        print("\nGenerating response...")
        try:
            response = chat_engine.chat(question)
            print("\n--- RESPONSE ---")
            print(response.response)
            print("----------------")
        except RateLimitError:
            if not api_keys or len(api_keys) == 1:
                print("\n--- ERROR ---", file=sys.stderr)
                print("The API key is rate-limited. Please try again later or add more keys.", file=sys.stderr)
                print("---------------", file=sys.stderr)
                continue

            current_key_index = (current_key_index + 1) % len(api_keys)
            new_key = api_keys[current_key_index]
            print(f"\n--- INFO: Rate limit hit. Switching to key index {current_key_index}. ---\n")
            
            # Update the LLM in the chat engine with the new key
            chat_engine._llm = OpenRouter(api_key=new_key, **original_llm_settings)
            
            # We retry the same question automatically in the next loop iteration if the user re-enters it.
            # For simplicity, we just notify the user. A more complex implementation would retry automatically.
            print("Please ask your question again to retry with the new key.")

        except Exception as e:
            print(f"\nAn error occurred during the query: {e}", file=sys.stderr)

def vectorbt_mode(api_mode=False):
    """
    Handles the logic for querying the VectorBT documentation and codebase.
    Returns a chat_engine.
    """
    setup_global_settings(max_tokens=2048)
    print("Loading VectorBT index from disk...")
    
    if not os.path.exists(CHROMA_PATH):
        print(f"Error: Index directory not found at {CHROMA_PATH}", file=sys.stderr)
        print("Please build the index first by running: python scripts/build_index.py", file=sys.stderr)
        if api_mode:
            raise FileNotFoundError(f"Index not found at {CHROMA_PATH}")
        else:
            sys.exit(1)

    try:
        # Recreate the ChromaVectorStore with the specific collection
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        chroma_collection = chroma_client.get_collection("vectorbt_docs") # Use get_collection to avoid recreating if it doesn't exist
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Load the storage context from disk, providing the vector_store
        storage_context = StorageContext.from_defaults(
            persist_dir=CHROMA_PATH,
            vector_store=vector_store
        )
        
        # Load the index from the fully configured storage context
        index = load_index_from_storage(storage_context=storage_context)
        
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            similarity_top_k=15,
            node_postprocessors=[
                CohereRerank(api_key=os.getenv("COHERE_API_KEY"), top_n=5)
            ],
        )
        
        if api_mode:
            return chat_engine

        print("\n" + "="*50)
        print(f"Assistant ready: VectorBT Assistant")
        print("Ask any question about VectorBT. Type 'exit' or 'quit' to return to the main menu.")
        print("="*50 + "\n")
        run_chat_loop(chat_engine, "VectorBT Assistant")
    except Exception as e:
        print(f"Error loading vector index: {e}", file=sys.stderr)
        sys.exit(1)

def review_mode(api_mode=False, code_snippet=None):
    """
    Handles the logic for reviewing a code snippet.
    Returns a chat_engine.
    """
    setup_global_settings(context_window=163840, max_tokens=4096)

    if not api_mode:
        code_to_review = get_multiline_input(
            "\nPaste your code below:"
        )
    else:
        code_to_review = code_snippet

    if not code_to_review or not code_to_review.strip():
        print("No code provided. Returning to main menu.", file=sys.stderr)
        return
    
    print("\nAnalyzing the code and building a temporary index...")
    try:
        code_document = Document(text=code_to_review)
        index = VectorStoreIndex.from_documents([code_document])
        
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            similarity_top_k=10,
            node_postprocessors=[
                CohereRerank(api_key=os.getenv("COHERE_API_KEY"), top_n=3)
            ],
        )

        if api_mode:
            return chat_engine

    except Exception as e:
        print(f"An error occurred during indexing: {e}", file=sys.stderr)
        return

    print("\n" + "="*50)
    print(f"Assistant ready: Code Review Assistant")
    print("Ask any question about the code. Type 'exit' or 'quit' to return to the main menu.")
    print("="*50 + "\n")
    run_chat_loop(chat_engine, "Code Review Assistant")

def main():
    """
    Main function to let the user choose a mode and run the assistant.
    """
    print(f"--- Unified RAG Assistant (Model: {OPENROUTER_MODEL}) ---")
    while True:
        print("\nPlease choose a mode:")
        print("  1: Chat with Vectorbt Documentation & Codebase")
        print("  2: Review a temporary code snippet")
        print("  exit: Quit the assistant")
        
        choice = input("Enter your choice (1, 2, or exit): ").strip().lower()

        if choice == '1':
            vectorbt_mode()
            break
        elif choice == '2':
            review_mode()
            break
        elif choice in ['exit', 'quit', 'q']:
            break
        else:
            print("Invalid choice. Please enter 1, 2, or exit.")

    print("\nAu revoir!")

if __name__ == "__main__":
    main() 
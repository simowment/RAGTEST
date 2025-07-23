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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from openai import RateLimitError
import chromadb
from llm_manager import LLMManager

# Load environment variables
load_dotenv()

# Constants
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")

def setup_global_settings(llm_manager, llm_settings_override=None):
    """
    Set up the global settings for LlamaIndex using the LLMManager.
    """
    llm_settings = {
        "temperature": 0.1,
        "max_tokens": 4096,
        "context_window": 163840,
        **(llm_settings_override or {}),
    }
    llm_manager.llm_settings = llm_settings
    
    # Set the initial LLM
    Settings.llm = llm_manager.get_llm()
    
    # Set other global settings
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
        
    if not os.getenv("COHERE_API_KEY"):
        print("Error: COHERE_API_KEY environment variable not set.", file=sys.stderr)
        print("Please get your key from https://dashboard.cohere.com/api-keys", file=sys.stderr)
        sys.exit(1)

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

def run_chat_loop(chat_engine, session_name, llm_manager):
    """
    Main chat loop for a given chat engine, now using LLMManager.
    """
    max_retries = len(llm_manager.configurations)

    while True:
        question = input(f"\nYour question for {session_name} (or 'exit' to quit): ")
        if question.strip().lower() in ['exit', 'quit', 'q']:
            break
        if not question.strip():
            continue

        print("\nGenerating response...")
        
        for attempt in range(max_retries):
            try:
                # Update engine's LLM to the current one from the manager
                chat_engine._llm = llm_manager.get_llm()
                response = chat_engine.chat(question)
                
                print("\n--- RESPONSE ---")
                print(response.response)
                print("----------------")
                break # Success, so break the retry loop
            
            except RateLimitError as e:
                print(f"--> Caught RateLimitError (Attempt {attempt + 1}/{max_retries}).")
                llm_manager.handle_rate_limit(e)
                llm_manager.switch_to_next_config()
                if attempt < max_retries - 1:
                    print("--> Retrying with new configuration...")
                else:
                    print("\n--- ERROR ---", file=sys.stderr)
                    print("All API key/model configurations failed after retries.", file=sys.stderr)
                    print("---------------", file=sys.stderr)
            
            except Exception as e:
                print(f"\nAn error occurred: {e}", file=sys.stderr)
                print("--> Switching configuration and retrying...")
                llm_manager.switch_to_next_config()
                if attempt >= max_retries -1:
                    print("\n--- ERROR ---", file=sys.stderr)
                    print("An unexpected error occurred and all configurations failed.", file=sys.stderr)
                    print("---------------", file=sys.stderr)

def vectorbt_mode(api_mode=False, llm_manager=None):
    """
    Handles the logic for querying the VectorBT documentation and codebase.
    Returns a chat_engine.
    """
    if not llm_manager:
        llm_manager = LLMManager()

    setup_global_settings(llm_manager, llm_settings_override={"max_tokens": 2048})
    print("Loading VectorBT index from disk...")
    
    if not os.path.exists(CHROMA_PATH):
        print(f"Error: Index directory not found at {CHROMA_PATH}", file=sys.stderr)
        print("Please build the index first by running: python scripts/build_index.py", file=sys.stderr)
        if api_mode:
            raise FileNotFoundError(f"Index not found at {CHROMA_PATH}")
        else:
            sys.exit(1)

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        chroma_collection = chroma_client.get_collection("vectorbt_docs")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        storage_context = StorageContext.from_defaults(
            persist_dir=CHROMA_PATH,
            vector_store=vector_store
        )
        
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
        run_chat_loop(chat_engine, "VectorBT Assistant", llm_manager)
    except Exception as e:
        print(f"Error loading vector index: {e}", file=sys.stderr)
        sys.exit(1)

def review_mode(api_mode=False, code_snippet=None, llm_manager=None):
    """
    Handles the logic for reviewing a code snippet.
    Returns a chat_engine.
    """
    if not llm_manager:
        llm_manager = LLMManager()

    setup_global_settings(llm_manager, llm_settings_override={"context_window": 163840, "max_tokens": 4096})

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
    run_chat_loop(chat_engine, "Code Review Assistant", llm_manager)

def main():
    """
    Main function to let the user choose a mode and run the assistant.
    """
    try:
        # Initialize one LLMManager for the entire session
        llm_manager = LLMManager()
        print(f"--- Unified RAG Assistant ---")
        print(f"Models available: {', '.join(llm_manager.models)}")
        
        while True:
            print("\nPlease choose a mode:")
            print("  1: Chat with Vectorbt Documentation & Codebase")
            print("  2: Review a temporary code snippet")
            print("  exit: Quit the assistant")
            
            choice = input("Enter your choice (1, 2, or exit): ").strip().lower()

            if choice == '1':
                vectorbt_mode(llm_manager=llm_manager)
                break 
            elif choice == '2':
                review_mode(llm_manager=llm_manager)
                break
            elif choice in ['exit', 'quit', 'q']:
                break
            else:
                print("Invalid choice. Please enter 1, 2, or exit.")

    except ValueError as e:
        print(f"\nFatal Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected fatal error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nAu revoir!")

if __name__ == "__main__":
    main() 
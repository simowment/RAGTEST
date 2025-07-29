#!/usr/bin/env python3
"""
Unified RAG assistant for Vectorbt documentation and on-the-fly code review.
"""

import os
import sys
from dotenv import load_dotenv

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
import chromadb
from llm_manager import LLMManager

# Load environment variables
load_dotenv()

# Constants
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma_db")
DOCS_PATH = os.getenv("DOCS_PATH", "data/docs_vbt_clean")


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
    Boucle de chat simplifiée avec gestion d'erreur basique.
    """
    while True:
        question = input(f"\nVotre question pour {session_name} (ou 'exit' pour quitter): ")
        if question.strip().lower() in ['exit', 'quit', 'q']:
            break
        if not question.strip():
            continue

        print("\nGénération de la réponse...")
        
        max_retries = len(llm_manager.configurations)
        
        for attempt in range(max_retries):
            try:
                chat_engine._llm = llm_manager.get_llm()
                response = chat_engine.chat(question)
                
                print("\n--- RÉPONSE ---")
                print(response.response)
                print("---------------")
                break
            
            except Exception as e:
                print(f"Tentative {attempt + 1}/{max_retries} échouée")
                
                if attempt < max_retries - 1:
                    if llm_manager.is_rate_limit_error(e):
                        print("Rate limit → modèle saturé, changement immédiat")
                    else:
                        print("Erreur → changement de configuration")
                    llm_manager.switch_to_next_config()
                else:
                    print(f"Erreur finale: {e}")
                    break

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

        if api_mode:
            return index

        # Create chat engine with reranking for CLI mode
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            similarity_top_k=15,
            node_postprocessors=[
                CohereRerank(api_key=os.getenv("COHERE_API_KEY"), top_n=5)
            ],
            llm=llm_manager.get_llm()
        )

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
    Handles the logic for reviewing a code snippet using direct LLM interaction.
    Returns a simple chat interface for code review.
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
    
    # Create a simple chat interface that includes the code in context
    class CodeReviewChat:
        """Simple chat interface for code review without RAG overhead."""
        
        def __init__(self, code, llm_manager):
            self.code = code
            self.llm_manager = llm_manager
            self.conversation_history = []
            
        def _build_context(self, question):
            """Build the full context including code and conversation history."""
            context_parts = [
                "You are a code review assistant. Please analyze the following code and answer questions about it.",
                f"\n--- CODE TO REVIEW ---\n{self.code}\n--- END CODE ---\n"
            ]
            
            # Add conversation history for context
            if self.conversation_history:
                context_parts.append("\n--- CONVERSATION HISTORY ---")
                for i, (q, a) in enumerate(self.conversation_history[-3:]):  # Keep last 3 exchanges
                    context_parts.append(f"Q{i+1}: {q}")
                    context_parts.append(f"A{i+1}: {a}")
                context_parts.append("--- END HISTORY ---\n")
            
            context_parts.append(f"Current question: {question}")
            return "\n".join(context_parts)
            
        def chat(self, question):
            """Synchronous chat method for CLI usage."""
            full_prompt = self._build_context(question)
            
            # Get response from LLM
            llm = self.llm_manager.get_llm()
            response = llm.complete(full_prompt)
            response_text = str(response)
            
            # Store in conversation history
            self.conversation_history.append((question, response_text))
            
            # Return response in expected format
            class SimpleResponse:
                def __init__(self, text):
                    self.response = text
            
            return SimpleResponse(response_text)
            
        async def achat(self, question):
            """Async chat method for API usage."""
            full_prompt = self._build_context(question)
            
            # Get response from LLM
            llm = self.llm_manager.get_llm()
            response = await llm.acomplete(full_prompt)
            response_text = str(response)
            
            # Store in conversation history
            self.conversation_history.append((question, response_text))
            
            # Return response in expected format
            class SimpleResponse:
                def __init__(self, text):
                    self.response = text
            
            return SimpleResponse(response_text)
    
    chat_engine = CodeReviewChat(code_to_review, llm_manager)

    if api_mode:
        return chat_engine

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
#!/usr/bin/env python3
"""
Unified RAG assistant supporting multiple knowledge bases and multimodal capabilities.
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
from knowledge_bases import KnowledgeBaseManager, KnowledgeBaseType

# Load environment variables
load_dotenv()

# Constants
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma/vectorbt_db")
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

def load_knowledge_base(kb_id: str, api_mode=False, llm_manager=None):
    """
    Load a knowledge base by ID and return the appropriate chat engine.
    """
    if not llm_manager:
        llm_manager = LLMManager()

    kb_manager = KnowledgeBaseManager()
    kb_config = kb_manager.get_knowledge_base(kb_id)
    
    if not kb_config:
        raise ValueError(f"Knowledge base '{kb_id}' not found")
    
    if kb_config.type == KnowledgeBaseType.CODE_REVIEW:
        # Code review mode doesn't use persistent storage
        return None  # Will be handled separately
    
    # Handle unified strategy assistant
    if kb_id == "unified_strategy":
        return load_unified_strategy_assistant(api_mode, llm_manager)
    
    setup_global_settings(llm_manager, llm_settings_override={"max_tokens": 2048})
    print(f"Loading {kb_config.name} index from disk...")
    
    # Check if database exists, auto-build if missing
    kb_manager = KnowledgeBaseManager()
    if not kb_manager.auto_build_knowledge_base(kb_id):
        error_msg = f"Error: Failed to build or find index for {kb_config.name}"
        print(error_msg, file=sys.stderr)
        if api_mode:
            raise FileNotFoundError(error_msg)
        else:
            sys.exit(1)
    
    # Use the configured path (should exist after auto-build)
    actual_path = kb_config.chroma_path

    try:
        chroma_client = chromadb.PersistentClient(path=actual_path)
        chroma_collection = chroma_client.get_collection(kb_config.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        storage_context = StorageContext.from_defaults(
            persist_dir=actual_path,
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
        print(f"Assistant ready: {kb_config.name}")
        print(f"Ask any question about {kb_config.description.lower()}. Type 'exit' or 'quit' to return to the main menu.")
        print("="*50 + "\n")
        run_chat_loop(chat_engine, kb_config.name, llm_manager)
    except Exception as e:
        print(f"Error loading vector index: {e}", file=sys.stderr)
        if api_mode:
            raise
        sys.exit(1)

def load_unified_strategy_assistant(api_mode=False, llm_manager=None):
    """
    Load the unified strategy assistant that combines VectorBT and Trading Papers.
    """
    if not llm_manager:
        llm_manager = LLMManager()

    setup_global_settings(llm_manager, llm_settings_override={"max_tokens": 4096})
    print("Loading Unified Strategy Assistant (VectorBT + Trading Papers)...")
    
    # Auto-build both knowledge bases if needed
    kb_manager = KnowledgeBaseManager()
    
    if not kb_manager.auto_build_knowledge_base("vectorbt"):
        error_msg = "Failed to build or find VectorBT index"
        if api_mode:
            raise FileNotFoundError(error_msg)
        print(error_msg, file=sys.stderr)
        sys.exit(1)
    
    if not kb_manager.auto_build_knowledge_base("trading_papers"):
        error_msg = "Failed to build or find Trading Papers index"
        if api_mode:
            raise FileNotFoundError(error_msg)
        print(error_msg, file=sys.stderr)
        sys.exit(1)
    
    # Use configured paths (should exist after auto-build)
    vectorbt_path = os.getenv("CHROMA_PATH", "data/chroma_db")
    trading_path = os.getenv("TRADING_CHROMA_PATH", "data/chroma/trading_db")

    try:
        # Load VectorBT index
        vectorbt_client = chromadb.PersistentClient(path=vectorbt_path)
        vectorbt_collection = vectorbt_client.get_collection("vectorbt_docs")
        vectorbt_vector_store = ChromaVectorStore(chroma_collection=vectorbt_collection)
        vectorbt_storage_context = StorageContext.from_defaults(
            persist_dir=vectorbt_path,
            vector_store=vectorbt_vector_store
        )
        vectorbt_index = load_index_from_storage(storage_context=vectorbt_storage_context)
        
        # Load Trading Papers index
        trading_client = chromadb.PersistentClient(path=trading_path)
        trading_collection = trading_client.get_collection("trading_papers_docs")
        trading_vector_store = ChromaVectorStore(chroma_collection=trading_collection)
        trading_storage_context = StorageContext.from_defaults(
            persist_dir=trading_path,
            vector_store=trading_vector_store
        )
        trading_index = load_index_from_storage(storage_context=trading_storage_context)

        if api_mode:
            # Return a unified chat engine for API mode
            return UnifiedStrategyChat(vectorbt_index, trading_index, llm_manager)

        # Create unified chat engine for CLI mode
        unified_chat = UnifiedStrategyChat(vectorbt_index, trading_index, llm_manager)
        
        print("\n" + "="*50)
        print("Assistant ready: Unified Strategy Assistant")
        print("Ask questions combining VectorBT technical knowledge with trading research insights.")
        print("="*50 + "\n")
        run_chat_loop(unified_chat, "Unified Strategy Assistant", llm_manager)
        
    except Exception as e:
        print(f"Error loading unified strategy assistant: {e}", file=sys.stderr)
        if api_mode:
            raise
        sys.exit(1)

class UnifiedStrategyChat:
    """Chat interface that queries both VectorBT and Trading Papers indices."""
    
    def __init__(self, vectorbt_index, trading_index, llm_manager):
        self.vectorbt_index = vectorbt_index
        self.trading_index = trading_index
        self.llm_manager = llm_manager
        self.conversation_history = []
        
        # Create retrieval engines
        self.vectorbt_retriever = vectorbt_index.as_retriever(similarity_top_k=8)
        self.trading_retriever = trading_index.as_retriever(similarity_top_k=7)
        
    def _retrieve_context(self, question):
        """Retrieve context from both knowledge bases."""
        # Get relevant nodes from both indices
        vectorbt_nodes = self.vectorbt_retriever.retrieve(question)
        trading_nodes = self.trading_retriever.retrieve(question)
        
        # Combine and format context
        context_parts = []
        
        if vectorbt_nodes:
            context_parts.append("=== VectorBT Technical Documentation ===")
            for i, node in enumerate(vectorbt_nodes[:5], 1):
                context_parts.append(f"VBT-{i}: {node.text}")
        
        if trading_nodes:
            context_parts.append("\n=== Trading Research Papers ===")
            for i, node in enumerate(trading_nodes[:5], 1):
                context_parts.append(f"PAPER-{i}: {node.text}")
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, question):
        """Build the full prompt with context and conversation history."""
        context = self._retrieve_context(question)
        
        prompt_parts = [
            "You are a unified strategy development assistant with access to both VectorBT technical documentation and trading research papers.",
            "Use the VectorBT documentation for technical implementation details and the trading papers for theoretical insights and strategy concepts.",
            "Provide comprehensive answers that combine both technical implementation and theoretical background.",
            "",
            "IMPORTANT: When showing code in your responses, ALWAYS use proper markdown code blocks with triple backticks (```) and specify the language:",
            "```python",
            "# Example code",
            "import vectorbt as vbt",
            "```",
            "For inline code, use single backticks like `vbt.Portfolio`.",
            "",
            "=== KNOWLEDGE BASE CONTEXT ===",
            context,
            "=== END CONTEXT ===",
            ""
        ]
        
        # Add conversation history
        if self.conversation_history:
            prompt_parts.append("=== CONVERSATION HISTORY ===")
            for i, (q, a) in enumerate(self.conversation_history[-2:], 1):
                prompt_parts.append(f"Q{i}: {q}")
                prompt_parts.append(f"A{i}: {a}")
            prompt_parts.append("=== END HISTORY ===\n")
        
        prompt_parts.append(f"Current question: {question}")
        
        return "\n".join(prompt_parts)
    
    def chat(self, question):
        """Synchronous chat method for CLI usage."""
        full_prompt = self._build_prompt(question)
        
        llm = self.llm_manager.get_llm()
        response = llm.complete(full_prompt)
        response_text = str(response)
        
        # Store in conversation history
        self.conversation_history.append((question, response_text))
        
        class SimpleResponse:
            def __init__(self, text):
                self.response = text
        
        return SimpleResponse(response_text)
    
    async def achat(self, question):
        """Async chat method for API usage."""
        full_prompt = self._build_prompt(question)
        
        llm = self.llm_manager.get_llm()
        response = await llm.acomplete(full_prompt)
        response_text = str(response)
        
        # Store in conversation history
        self.conversation_history.append((question, response_text))
        
        class SimpleResponse:
            def __init__(self, text):
                self.response = text
        
        return SimpleResponse(response_text)

def vectorbt_mode(api_mode=False, llm_manager=None):
    """
    Handles the logic for querying the VectorBT documentation and codebase.
    Returns a chat_engine.
    """
    return load_knowledge_base("vectorbt", api_mode, llm_manager)

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
                "",
                "IMPORTANT: When showing code in your responses, ALWAYS use proper markdown code blocks with triple backticks (```) and specify the language. For example:",
                "```python",
                "def example():",
                "    return 'code here'",
                "```",
                "",
                "For inline code references, use single backticks like `variable_name`.",
                "",
                f"--- CODE TO REVIEW ---\n{self.code}\n--- END CODE ---"
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
    Main function to let the user choose a knowledge base and run the assistant.
    """
    try:
        # Initialize managers
        llm_manager = LLMManager()
        kb_manager = KnowledgeBaseManager()
        
        print(f"--- Multi-Modal RAG Assistant ---")
        print(f"Models available: {', '.join(llm_manager.models)}")
        
        while True:
            print("\nAvailable Knowledge Bases:")
            available_kbs = kb_manager.get_available_knowledge_bases()
            
            for i, kb in enumerate(available_kbs, 1):
                status = "✅" if kb_manager.knowledge_base_exists(kb.id) else "❌"
                print(f"  {i}: {kb.icon} {kb.name} {status}")
                print(f"     {kb.description}")
                if kb.supports_images:
                    print(f"     📷 Supports images")
            
            print("  exit: Quit the assistant")
            
            choice = input(f"\nEnter your choice (1-{len(available_kbs)}, or exit): ").strip().lower()

            if choice in ['exit', 'quit', 'q']:
                break
            
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_kbs):
                    selected_kb = available_kbs[choice_idx]
                    
                    if not kb_manager.knowledge_base_exists(selected_kb.id):
                        print(f"❌ Knowledge base '{selected_kb.name}' is not available.")
                        print("Please build the index first or check the configuration.")
                        continue
                    
                    if selected_kb.type == KnowledgeBaseType.CODE_REVIEW:
                        review_mode(llm_manager=llm_manager)
                    else:
                        load_knowledge_base(selected_kb.id, llm_manager=llm_manager)
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid choice. Please enter a number or 'exit'.")

    except ValueError as e:
        print(f"\nFatal Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected fatal error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nAu revoir!")

if __name__ == "__main__":
    main() 
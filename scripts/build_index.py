#!/usr/bin/env python3
"""
Script to build a vector index from the text documents in docs_vbt_clean
and the vectorbt Python codebase.
Using llama-index and chromadb.
"""

import os
import sys
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Load environment variables
load_dotenv()

# Constants
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
DOCS_PATH = os.getenv("DOCS_PATH", "docs_vbt_clean")
CODE_PATH = os.getenv("VECTORBT_CODEBASE_PATH")

def build_index(documents, persist_dir):
    """
    Build and persist a ChromaDB vector index using llama-index
    """
    print(f"Building and persisting index to {persist_dir}...")
    
    # Initialize ChromaDB client and specify the collection
    db_path = os.path.join(os.getcwd(), persist_dir)
    chroma_client = chromadb.PersistentClient(path=db_path)
    # Ensure the collection name matches what is loaded in assistant.py
    chroma_collection = chroma_client.get_or_create_collection("vectorbt_docs") 
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create StorageContext with the specific vector_store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Build the index using the provided documents and storage context
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    # Persist the whole index (this will save Chroma's data via storage_context)
    index.storage_context.persist(persist_dir=persist_dir)
    print("Index built successfully.")

def main():
    """
    Main function to build the vector index
    """
    print("Building vector index for Vectorbt documentation and codebase...")

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    all_documents = []

    # 1. Load documentation files
    if os.path.exists(DOCS_PATH):
        print(f"Loading documentation from '{DOCS_PATH}'...")
        docs_reader = SimpleDirectoryReader(
            input_dir=DOCS_PATH,
            required_exts=[".txt"],
            recursive=True
        )
        docs = docs_reader.load_data()
        all_documents.extend(docs)
        print(f"Loaded {len(docs)} documentation files.")
    else:
        print(f"Warning: Documentation directory '{DOCS_PATH}' not found. Skipping.")

    # 2. Load codebase files
    if CODE_PATH:
        code_main_path = os.path.join(CODE_PATH, "vectorbt")
        if os.path.exists(code_main_path):
            print(f"Loading codebase from '{code_main_path}'...")
            code_reader = SimpleDirectoryReader(
                input_dir=code_main_path,
                required_exts=[".py"],
                recursive=True,
                exclude_hidden=True,
            )
            code_docs = code_reader.load_data()
            all_documents.extend(code_docs)
            print(f"Loaded {len(code_docs)} code files.")
        else:
            print(f"Warning: Code directory '{code_main_path}' not found inside '{CODE_PATH}'. Skipping codebase indexing.")
    else:
        print("Info: VECTORBT_CODEBASE_PATH environment variable not set. Skipping codebase indexing.")

    if not all_documents:
        print("Error: No documents or code files found to index. Aborting.")
        sys.exit(1)

    print(f"\nTotal documents to be indexed: {len(all_documents)}")
    
    index = build_index(all_documents, CHROMA_PATH)
    
    print("\nIndex building complete! You can now query the documents and codebase.")
    print(f"Run: python scripts/rag_query.py")

if __name__ == "__main__":
    main() 
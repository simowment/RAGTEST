#!/usr/bin/env python3
"""
Script to build vector indices for multiple knowledge bases.
Supports VectorBT documentation, trading papers, and more.
Using llama-index and chromadb.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

# Knowledge base configurations
KNOWLEDGE_BASES = {
    "vectorbt": {
        "name": "VectorBT Documentation & Codebase",
        "chroma_path": os.getenv("CHROMA_PATH", "data/chroma/vectorbt_db"),
        "collection_name": "vectorbt_docs",
        "sources": [
            {
                "path": os.getenv("DOCS_PATH", "data/docs_vbt_clean"),
                "extensions": [".txt"],
                "description": "VectorBT documentation"
            },
            {
                "path": os.path.dirname(os.getenv("DOCS_PATH", "data/docs_vbt_clean")),
                "extensions": [".py"],
                "description": "Python files from data directory"
            }
        ],
        "additional_code_path": os.getenv("VECTORBT_CODEBASE_PATH")
    },
    "trading_papers": {
        "name": "Trading Research Papers",
        "chroma_path": os.getenv("TRADING_CHROMA_PATH", "data/chroma/trading_db"),
        "collection_name": "trading_papers_docs",
        "sources": [
            {
                "path": os.getenv("TRADING_DOCS_PATH", "data/trading_papers"),
                "extensions": [".pdf", ".txt", ".md", ".docx"],
                "description": "Trading research papers"
            }
        ]
    }
}

def setup_settings():
    """Configure global settings for LlamaIndex."""
    try:
        # Use optimal chunk size for PDF content with long file paths
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            max_length=512  # Ensure we don't exceed model limits
        )
        Settings.chunk_size = 512  # Larger chunks to accommodate metadata
        Settings.chunk_overlap = 50
        print("✅ Configured HuggingFace embeddings with robust settings")
    except Exception as e:
        print(f"❌ Error setting up embeddings: {e}")
        raise

def build_index(documents, persist_dir, collection_name):
    """
    Build and persist a ChromaDB vector index using llama-index
    """
    print(f"Building and persisting index to {persist_dir}...")
    
    # Ensure the directory exists
    os.makedirs(persist_dir, exist_ok=True)
    
    # Initialize ChromaDB client and specify the collection
    db_path = os.path.join(os.getcwd(), persist_dir)
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_or_create_collection(collection_name) 
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

def load_documents_from_source(source_config):
    """Load documents from a single source configuration."""
    path = source_config["path"]
    extensions = source_config["extensions"]
    description = source_config["description"]
    
    if not os.path.exists(path):
        print(f"Warning: {description} directory '{path}' not found. Skipping.")
        return []
    
    print(f"Loading {description} from '{path}'...")
    try:
        reader = SimpleDirectoryReader(
            input_dir=path,
            required_exts=extensions,
            recursive=True,
            exclude_hidden=True,
        )
        documents = reader.load_data()
        
        # Filter and clean documents
        valid_documents = []
        for doc in documents:
            try:
                # Check if document has valid text content
                if hasattr(doc, 'text') and doc.text and isinstance(doc.text, str) and doc.text.strip():
                    # Clean the text more thoroughly
                    text = doc.text.strip()
                    
                    # Skip documents with null bytes
                    if '\x00' in text:
                        print(f"Warning: Skipping document with null bytes")
                        continue
                    
                    # Skip documents that are too short or too long
                    if len(text) < 10:
                        print(f"Warning: Skipping document too short")
                        continue
                    
                    if len(text) > 1000000:  # 1MB limit
                        print(f"Warning: Truncating very large document")
                        text = text[:1000000]
                    
                    # Clean up common PDF extraction artifacts
                    import re
                    
                    # Remove image placeholders and artifacts
                    text = re.sub(r'<image[^>]*>', '', text, flags=re.IGNORECASE)
                    text = re.sub(r'\[image[^\]]*\]', '', text, flags=re.IGNORECASE)
                    text = re.sub(r'Figure \d+[^\n]*\n', '', text)
                    text = re.sub(r'Table \d+[^\n]*\n', '', text)
                    
                    # Clean up LaTeX artifacts
                    text = re.sub(r'\$[^$]*\$', '[MATH]', text)  # Replace inline math
                    text = re.sub(r'\$\$[^$]*\$\$', '[EQUATION]', text)  # Replace display math
                    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)  # Remove LaTeX commands
                    text = re.sub(r'\\[a-zA-Z]+', '', text)  # Remove simple LaTeX commands
                    
                    # Remove excessive whitespace and control characters
                    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)  # Remove control chars
                    
                    # Remove lines that are mostly non-alphabetic (likely artifacts)
                    lines = text.split('\n')
                    clean_lines = []
                    for line in lines:
                        line = line.strip()
                        if len(line) > 5:  # Only check lines with some content
                            alpha_ratio = sum(c.isalpha() for c in line) / len(line)
                            if alpha_ratio > 0.3:  # At least 30% alphabetic characters
                                clean_lines.append(line)
                        elif len(line) > 0:
                            clean_lines.append(line)
                    
                    text = '\n'.join(clean_lines).strip()
                    
                    # Final validation
                    if len(text) < 50:  # Skip very short documents after cleaning
                        print(f"Warning: Skipping document too short after cleaning")
                        continue
                    
                    # Check for valid UTF-8 encoding
                    try:
                        text.encode('utf-8')
                    except UnicodeEncodeError:
                        print(f"Warning: Skipping document with encoding issues")
                        continue
                    
                    # Check if text is mostly readable (not binary artifacts)
                    printable_ratio = sum(c.isprintable() or c.isspace() for c in text) / len(text)
                    if printable_ratio < 0.8:
                        print(f"Warning: Skipping document with too many non-printable characters")
                        continue
                    
                    # Final check: ensure text doesn't contain problematic sequences
                    # that could break the tokenizer
                    problematic_patterns = [
                        r'[\x00-\x08\x0B\x0C\x0E-\x1F]',  # Control characters
                        r'[\uFFFE\uFFFF]',  # Invalid Unicode
                        r'[\uD800-\uDFFF]',  # Surrogate pairs (should be paired)
                    ]
                    
                    for pattern in problematic_patterns:
                        if re.search(pattern, text):
                            print(f"Warning: Skipping document with problematic characters")
                            text = None
                            break
                    
                    if text is None:
                        continue
                    
                    # Create a new document with cleaned text
                    from llama_index.core import Document
                    cleaned_doc = Document(
                        text=text,
                        metadata=doc.metadata if hasattr(doc, 'metadata') else {}
                    )
                    valid_documents.append(cleaned_doc)
                    
            except Exception as e:
                print(f"Warning: Skipping problematic document: {e}")
                continue
        
        print(f"✅ Loaded {len(valid_documents)} valid files from {description}")
        return valid_documents
    except Exception as e:
        print(f"Warning: Could not load {description} from '{path}': {e}")
        return []

def build_knowledge_base(kb_id):
    """Build a specific knowledge base."""
    # Always setup settings to ensure proper configuration
    setup_settings()
    
    if kb_id not in KNOWLEDGE_BASES:
        print(f"❌ Error: Unknown knowledge base '{kb_id}'")
        print(f"Available knowledge bases: {', '.join(KNOWLEDGE_BASES.keys())}")
        return False
    
    kb_config = KNOWLEDGE_BASES[kb_id]
    print(f"\n{'='*60}")
    print(f"Building {kb_config['name']}")
    print(f"{'='*60}")
    
    all_documents = []
    
    # Load documents from all sources
    for source in kb_config["sources"]:
        documents = load_documents_from_source(source)
        all_documents.extend(documents)
    
    # Handle additional code path for VectorBT
    if kb_id == "vectorbt" and kb_config.get("additional_code_path"):
        code_path = kb_config["additional_code_path"]
        if os.path.exists(code_path):
            vectorbt_path = os.path.join(code_path, "vectorbt")
            if os.path.exists(vectorbt_path):
                print(f"Loading additional VectorBT codebase from '{vectorbt_path}'...")
                try:
                    code_reader = SimpleDirectoryReader(
                        input_dir=vectorbt_path,
                        required_exts=[".py"],
                        recursive=True,
                        exclude_hidden=True,
                    )
                    additional_docs = code_reader.load_data()
                    all_documents.extend(additional_docs)
                    print(f"✅ Loaded {len(additional_docs)} additional code files")
                except Exception as e:
                    print(f"Warning: Could not load from '{vectorbt_path}': {e}")
            else:
                print(f"Warning: vectorbt subdirectory not found in '{code_path}'")
        else:
            print(f"Warning: VECTORBT_CODEBASE_PATH '{code_path}' does not exist")
    
    if not all_documents:
        print("❌ Error: No documents found to index. Aborting.")
        return False
    
    print(f"\n📊 Total documents to be indexed: {len(all_documents)}")
    
    # Build the index
    try:
        build_index(
            all_documents, 
            kb_config["chroma_path"], 
            kb_config["collection_name"]
        )
        
        print(f"\n✅ {kb_config['name']} index built successfully!")
        print(f"📁 Index saved to: {kb_config['chroma_path']}")
        print(f"🔍 Collection: {kb_config['collection_name']}")
        return True
        
    except Exception as e:
        print(f"❌ Error building index: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to build vector indices for knowledge bases.
    """
    parser = argparse.ArgumentParser(
        description="Build vector indices for knowledge bases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_index.py vectorbt           # Build VectorBT index
  python scripts/build_index.py trading_papers     # Build trading papers index
  python scripts/build_index.py --all              # Build all available indices
  python scripts/build_index.py --list             # List available knowledge bases
        """
    )
    
    parser.add_argument(
        "knowledge_base", 
        nargs="?",
        help="Knowledge base to build (vectorbt, trading_papers)"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Build all available knowledge bases"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List available knowledge bases"
    )
    
    args = parser.parse_args()
    
    # Setup global settings
    setup_settings()
    
    if args.list:
        print("Available Knowledge Bases:")
        print("=" * 40)
        for kb_id, config in KNOWLEDGE_BASES.items():
            print(f"🔹 {kb_id}: {config['name']}")
            print(f"   Path: {config['chroma_path']}")
            print(f"   Collection: {config['collection_name']}")
            print()
        return
    
    if args.all:
        print("🚀 Building all knowledge bases...")
        success_count = 0
        for kb_id in KNOWLEDGE_BASES.keys():
            if build_knowledge_base(kb_id):
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"✅ Successfully built {success_count}/{len(KNOWLEDGE_BASES)} knowledge bases")
        print("🚀 You can now use all knowledge bases in the web interface!")
        return
    
    if not args.knowledge_base:
        print("❌ Error: Please specify a knowledge base to build or use --all")
        print("Use --list to see available knowledge bases")
        parser.print_help()
        sys.exit(1)
    
    # Build specific knowledge base
    if build_knowledge_base(args.knowledge_base):
        print(f"\n🚀 You can now use the '{args.knowledge_base}' knowledge base!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 
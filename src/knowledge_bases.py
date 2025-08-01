#!/usr/bin/env python3
"""
Knowledge base configuration and management system.
Supports multiple data sources and modes.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class KnowledgeBaseType(Enum):
    VECTORBT = "vectorbt"
    TRADING_PAPERS = "trading_papers"
    CODE_REVIEW = "code_review"

@dataclass
class KnowledgeBaseConfig:
    """Configuration for a knowledge base."""
    id: str
    name: str
    description: str
    type: KnowledgeBaseType
    chroma_path: str
    docs_path: Optional[str] = None
    collection_name: str = None
    supports_images: bool = False
    icon: str = "📚"
    
    def __post_init__(self):
        if self.collection_name is None:
            self.collection_name = f"{self.id}_docs"

class KnowledgeBaseManager:
    """Manages multiple knowledge bases and their configurations."""
    
    def __init__(self):
        self.knowledge_bases = self._load_knowledge_bases()
        
    def _load_knowledge_bases(self) -> Dict[str, KnowledgeBaseConfig]:
        """Load all available knowledge bases."""
        bases = {}
        
        # VectorBT Documentation
        vectorbt_config = KnowledgeBaseConfig(
            id="vectorbt",
            name="VectorBT Documentation",
            description="Documentation et codebase VectorBT pour l'analyse quantitative",
            type=KnowledgeBaseType.VECTORBT,
            chroma_path=os.getenv("CHROMA_PATH", "data/chroma/vectorbt_db"),
            docs_path=os.getenv("DOCS_PATH", "data/vectorbt/docs_vbt_clean"),
            collection_name="vectorbt_docs",
            supports_images=True,
            icon="📊"
        )
        bases[vectorbt_config.id] = vectorbt_config
        
        # Trading Papers
        trading_config = KnowledgeBaseConfig(
            id="trading_papers",
            name="Trading Research Papers",
            description="Collection de papers de recherche sur le trading et la finance quantitative",
            type=KnowledgeBaseType.TRADING_PAPERS,
            chroma_path=os.getenv("TRADING_CHROMA_PATH", "data/chroma/trading_db"),
            docs_path=os.getenv("TRADING_DOCS_PATH", "data/trading_papers"),
            collection_name="trading_papers_docs",
            supports_images=True,
            icon="📈"
        )
        bases[trading_config.id] = trading_config
        
        # Unified Strategy Assistant (queries both VectorBT and Trading Papers)
        unified_config = KnowledgeBaseConfig(
            id="unified_strategy",
            name="Unified Strategy Assistant",
            description="Assistant combinant VectorBT et papers de trading pour développer des stratégies complètes",
            type=KnowledgeBaseType.VECTORBT,  # Uses VectorBT type but special handling
            chroma_path="",  # Special handling - uses multiple sources
            supports_images=True,
            icon="🚀"
        )
        bases[unified_config.id] = unified_config
        
        # Code Review (special case - no persistent storage)
        code_review_config = KnowledgeBaseConfig(
            id="code_review",
            name="Code Review Assistant",
            description="Assistant pour la review de code avec support d'images",
            type=KnowledgeBaseType.CODE_REVIEW,
            chroma_path="",  # No persistent storage
            supports_images=True,
            icon="🔍"
        )
        bases[code_review_config.id] = code_review_config
        
        return bases
    
    def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBaseConfig]:
        """Get a knowledge base configuration by ID."""
        return self.knowledge_bases.get(kb_id)
    
    def get_available_knowledge_bases(self) -> List[KnowledgeBaseConfig]:
        """Get all available knowledge bases."""
        return list(self.knowledge_bases.values())
    
    def get_knowledge_bases_with_images(self) -> List[KnowledgeBaseConfig]:
        """Get knowledge bases that support images."""
        return [kb for kb in self.knowledge_bases.values() if kb.supports_images]
    
    def knowledge_base_exists(self, kb_id: str) -> bool:
        """Check if a knowledge base exists and is accessible."""
        config = self.get_knowledge_base(kb_id)
        if not config:
            return False
            
        if config.type == KnowledgeBaseType.CODE_REVIEW:
            return True  # Code review doesn't need persistent storage
        
        if kb_id == "unified_strategy":
            # Unified strategy needs both VectorBT and Trading Papers
            vectorbt_path = os.getenv("CHROMA_PATH", "data/chroma_db")
            trading_path = os.getenv("TRADING_CHROMA_PATH", "data/chroma/trading_db")
            return os.path.exists(vectorbt_path) and os.path.exists(trading_path)
            
        return os.path.exists(config.chroma_path)
    
    def auto_build_knowledge_base(self, kb_id: str) -> bool:
        """Automatically build a knowledge base if it doesn't exist."""
        if self.knowledge_base_exists(kb_id):
            return True
            
        config = self.get_knowledge_base(kb_id)
        if not config or config.type == KnowledgeBaseType.CODE_REVIEW:
            return True
            
        print(f"Knowledge base '{kb_id}' not found. Building automatically...")
        
        try:
            # Ensure proper embedding setup before building
            from llama_index.core import Settings
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            
            # Force HuggingFace embeddings
            Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            Settings.chunk_size = 512
            Settings.chunk_overlap = 50
            
            from scripts.build_index import build_knowledge_base
            return build_knowledge_base(kb_id)
        except Exception as e:
            print(f"Failed to auto-build knowledge base '{kb_id}': {e}")
            return False
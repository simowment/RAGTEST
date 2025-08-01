#!/usr/bin/env python3
"""
Gestionnaire LLM simplifié pour OpenRouter.
Gère les rate limits et les modèles défaillants.
"""

import os
import time
from itertools import cycle
from dotenv import load_dotenv
from llama_index.llms.openrouter import OpenRouter
from llama_index.postprocessor.cohere_rerank import CohereRerank
from openai import RateLimitError

load_dotenv()

class LLMManager:
    """Gestionnaire LLM simplifié avec fallback automatique."""

    def __init__(self, llm_settings=None):
        self.api_keys = self._load_api_keys()
        self.models = self._load_models()
        self.llm_settings = llm_settings or {}

        if not self.api_keys or not self.models:
            raise ValueError("Clés API ou modèles manquants dans .env")

        # Configurations disponibles
        self.configurations = [(key, model) for model in self.models for key in self.api_keys]
        self.config_cycler = cycle(self.configurations)
        self.current_config = next(self.config_cycler)

        print(f"LLM Manager: {len(self.configurations)} configurations disponibles")

    def _load_api_keys(self):
        """Charge les clés API depuis les variables d'environnement."""
        keys = []
        i = 1
        while True:
            key = os.getenv(f"OPENROUTER_API_KEY_{i}")
            if not key:
                break
            key = key.strip()
            if key and "YOUR_" not in key.upper() and key not in keys:
                keys.append(key)
            i += 1
        return keys

    def _load_models(self):
        """Charge la liste des modèles depuis OPENROUTER_MODELS."""
        models_str = os.getenv("OPENROUTER_MODELS")
        if models_str:
            return [m.strip() for m in models_str.split(',') if m.strip()]
        return []

    def get_llm(self):
        """Retourne une instance OpenRouter avec la config actuelle."""
        api_key, model_name = self.current_config
        
        settings = {
            "temperature": 0.1,
            "max_tokens": 4096,
            "context_window": 163840,
            **self.llm_settings,
            "model": model_name,
            "api_key": api_key,
        }
        return OpenRouter(**settings)

    def switch_to_next_config(self):
        """Passe à la configuration suivante."""
        self.current_config = next(self.config_cycler)
        
    def is_rate_limit_error(self, error):
        """Vérifie si l'erreur est due à un rate limit."""
        if isinstance(error, RateLimitError):
            return True
        
        error_str = str(error).lower()
        rate_limit_indicators = [
            "rate limit", "too many requests", "quota exceeded",
            "429", "rate_limit_exceeded"
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)
    
    def is_model_error(self, error):
        """Vérifie si l'erreur est due au modèle (indisponible, etc.)."""
        error_str = str(error).lower()
        model_error_indicators = [
            "model not found", "model unavailable", "invalid model",
            "model error", "service unavailable", "502", "503"
        ]
        return any(indicator in error_str for indicator in model_error_indicators)

class EnhancedChatWrapper:
    """Wrapper qui ajoute les instructions de formatage de code à tous les assistants."""
    
    def __init__(self, chat_engine, conversation_history=None):
        self.chat_engine = chat_engine
        self.conversation_history = conversation_history or []
    
    def _enhance_question(self, question):
        """Ajoute les instructions de formatage et l'historique à la question."""
        enhanced_parts = [
            "IMPORTANT: When showing code in your responses, ALWAYS use proper markdown code blocks with triple backticks (```) and specify the language:",
            "```python",
            "# Example code",
            "import vectorbt as vbt",
            "```",
            "For inline code, use single backticks like `vbt.Portfolio`.",
            ""
        ]
        
        # Ajouter l'historique de conversation si disponible
        if self.conversation_history:
            enhanced_parts.append("=== CONVERSATION HISTORY ===")
            for i, (q, a) in enumerate(self.conversation_history[-3:], 1):  # Derniers 3 échanges
                enhanced_parts.append(f"Q{i}: {q}")
                enhanced_parts.append(f"A{i}: {a}")
            enhanced_parts.append("=== END HISTORY ===")
            enhanced_parts.append("")
        
        enhanced_parts.append(f"Current question: {question}")
        
        return "\n".join(enhanced_parts)
    
    async def achat(self, question):
        """Chat asynchrone avec instructions de formatage."""
        enhanced_question = self._enhance_question(question)
        response = await self.chat_engine.achat(enhanced_question)
        
        # Stocker dans l'historique
        self.conversation_history.append((question, response.response))
        
        return response
    
    def chat(self, question):
        """Chat synchrone avec instructions de formatage."""
        enhanced_question = self._enhance_question(question)
        response = self.chat_engine.chat(enhanced_question)
        
        # Stocker dans l'historique
        self.conversation_history.append((question, response.response))
        
        return response

async def managed_chat_request(source, question, llm_manager):
    """
    Handles a chat request with automatic fallback and retry logic.
    This function creates a new chat engine for each request to avoid state issues.
    """
    print(f"🔍 [DEBUG] managed_chat_request called")
    print(f"🔍 [DEBUG] Source type: {type(source).__name__}")
    print(f"🔍 [DEBUG] Has as_chat_engine: {hasattr(source, 'as_chat_engine')}")
    print(f"🔍 [DEBUG] Available configurations: {len(llm_manager.configurations)}")
    
    max_retries = len(llm_manager.configurations)

    for attempt in range(max_retries):
        try:
            print(f"🔍 [DEBUG] Attempt {attempt + 1}/{max_retries} with config: {llm_manager.current_config}")
            llm = llm_manager.get_llm()

            # Determine the type of the source and create the appropriate chat engine
            if hasattr(source, 'as_chat_engine'):  # It's a VectorStoreIndex
                print(f"🔍 [DEBUG] Creating standard chat engine with reranking")
                # Créer le chat engine avec ou sans reranking selon la disponibilité de Cohere
                cohere_key = os.getenv("COHERE_API_KEY")
                if cohere_key and cohere_key.strip():
                    try:
                        base_chat_engine = source.as_chat_engine(
                            chat_mode="context",
                            similarity_top_k=15,
                            node_postprocessors=[
                                CohereRerank(api_key=cohere_key, top_n=5)
                            ],
                            llm=llm
                        )
                        print(f"🔍 [DEBUG] Cohere reranking enabled")
                    except Exception as e:
                        print(f"🔍 [DEBUG] Warning: Could not use Cohere reranking: {e}")
                        base_chat_engine = source.as_chat_engine(
                            chat_mode="context",
                            similarity_top_k=10,
                            llm=llm
                        )
                        print(f"🔍 [DEBUG] Using fallback without reranking")
                else:
                    base_chat_engine = source.as_chat_engine(
                        chat_mode="context",
                        similarity_top_k=10,
                        llm=llm
                    )
                    print(f"🔍 [DEBUG] No Cohere key, using basic engine")
                # Wrapper avec instructions de formatage
                chat_engine = EnhancedChatWrapper(base_chat_engine)
                print(f"🔍 [DEBUG] Wrapped with EnhancedChatWrapper")
            else:  # It's a custom chat object like CodeReviewChat
                print(f"🔍 [DEBUG] Using custom chat engine: {type(source).__name__}")
                chat_engine = source
                # Here, we assume the custom chat object will use the llm_manager to get the llm.

            response = await chat_engine.achat(question)
            print(f"🔍 [DEBUG] Chat response received, length: {len(response.response)}")
            return {"response": response.response}

        except Exception as e:
            print(f"🔍 [DEBUG] Attempt {attempt + 1}/{max_retries} failed: {str(e)[:100]}")

            if attempt < max_retries - 1:
                if llm_manager.is_rate_limit_error(e) or llm_manager.is_model_error(e):
                    print("🔍 [DEBUG] Rate limit or model error detected, switching configuration.")
                    llm_manager.switch_to_next_config()
                else:
                    print("🔍 [DEBUG] Unknown error, switching configuration as a precaution.")
                    llm_manager.switch_to_next_config()
            else:
                # Last attempt failed
                print(f"🔍 [DEBUG] All attempts failed, raising exception")
                raise e

    raise Exception("All LLM configurations failed.")
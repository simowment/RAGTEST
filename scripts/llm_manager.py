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

async def managed_chat_request(chat_engine, question, llm_manager):
    """
    Fonction simplifiée pour gérer les requêtes avec fallback automatique.
    """
    max_retries = len(llm_manager.configurations)
    
    for attempt in range(max_retries):
        try:
            # Utilise le LLM actuel
            chat_engine._llm = llm_manager.get_llm()
            response = await chat_engine.achat(question)
            return {"response": str(response)}
        
        except Exception as e:
            print(f"Tentative {attempt + 1}/{max_retries} échouée: {str(e)[:100]}")
            
            if attempt < max_retries - 1:
                # Rate limit : modèle saturé, changer immédiatement
                if llm_manager.is_rate_limit_error(e):
                    print("Rate limit détecté, modèle saturé → changement de configuration")
                    llm_manager.switch_to_next_config()
                    continue
                
                # Erreur de modèle : changer immédiatement
                elif llm_manager.is_model_error(e):
                    print("Erreur de modèle → changement de configuration")
                    llm_manager.switch_to_next_config()
                    continue
                
                # Autres erreurs : essayer une autre config
                else:
                    print("Erreur inconnue → changement de configuration")
                    llm_manager.switch_to_next_config()
                    continue
            
            # Dernière tentative échouée
            if attempt == max_retries - 1:
                raise e
    
    raise Exception("Toutes les configurations ont échoué") 
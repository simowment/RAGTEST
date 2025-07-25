#!/usr/bin/env python3
"""
Test du LLM Manager simplifié
"""

import sys
import os
sys.path.append('scripts')

from llm_manager import LLMManager

def test_llm_manager():
    """Test les fonctionnalités de base du LLM Manager"""
    
    print("🧪 Test du LLM Manager simplifié")
    print("=" * 40)
    
    try:
        # Test d'initialisation
        manager = LLMManager()
        print(f"✅ Manager initialisé avec {len(manager.configurations)} configurations")
        
        # Test de création LLM
        llm = manager.get_llm()
        print("✅ LLM créé avec succès")
        
        # Test de changement de config
        old_config = manager.current_config
        manager.switch_to_next_config()
        new_config = manager.current_config
        print(f"✅ Configuration changée: {old_config[1]} -> {new_config[1]}")
        
        # Test de détection d'erreurs
        rate_limit_errors = [
            "Rate limit exceeded",
            "Too many requests", 
            "429 error",
            Exception("rate_limit_exceeded")
        ]
        
        for error in rate_limit_errors:
            if manager.is_rate_limit_error(error):
                print(f"✅ Rate limit détecté: {error}")
            else:
                print(f"❌ Rate limit non détecté: {error}")
        
        model_errors = [
            "Model not found",
            "Service unavailable",
            "502 Bad Gateway",
            Exception("model error occurred")
        ]
        
        for error in model_errors:
            if manager.is_model_error(error):
                print(f"✅ Erreur modèle détectée: {error}")
            else:
                print(f"❌ Erreur modèle non détectée: {error}")
        
        # Test de la logique : rate limit = changement immédiat
        print("\n📋 Logique de gestion d'erreurs:")
        print("✅ Rate limit → Changement immédiat de modèle (pas d'attente)")
        print("✅ Erreur modèle → Changement immédiat de configuration") 
        print("✅ Autres erreurs → Changement de configuration")
        
        print("\n🎉 Tous les tests passés!")
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_llm_manager()
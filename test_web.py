#!/usr/bin/env python3
"""
Script de test pour vérifier que l'interface web fonctionne correctement
"""

import requests
import json
import time

def test_api():
    """Test les endpoints de l'API"""
    base_url = "http://localhost:8000"
    
    print("🧪 Test de l'API RAG Assistant")
    print("=" * 40)
    
    # Test 1: Page d'accueil
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            print("✅ Page d'accueil accessible")
        else:
            print(f"❌ Page d'accueil: {response.status_code}")
    except Exception as e:
        print(f"❌ Erreur page d'accueil: {e}")
        return False
    
    # Test 2: Documentation VectorBT
    try:
        response = requests.post(
            f"{base_url}/vectorbt/query",
            json={"question": "Qu'est-ce que VectorBT?"},
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print("✅ Endpoint VectorBT fonctionne")
            print(f"   Réponse: {result.get('response', 'N/A')[:100]}...")
        else:
            print(f"❌ VectorBT endpoint: {response.status_code}")
            print(f"   Erreur: {response.text}")
    except Exception as e:
        print(f"❌ Erreur VectorBT: {e}")
    
    # Test 3: Review de code
    try:
        test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        response = requests.post(
            f"{base_url}/review/code",
            json={
                "code": test_code,
                "question": "Comment optimiser cette fonction?"
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print("✅ Endpoint Review fonctionne")
            print(f"   Réponse: {result.get('response', 'N/A')[:100]}...")
        else:
            print(f"❌ Review endpoint: {response.status_code}")
            print(f"   Erreur: {response.text}")
    except Exception as e:
        print(f"❌ Erreur Review: {e}")
    
    print("\n🎉 Tests terminés!")
    return True

if __name__ == "__main__":
    print("Assurez-vous que le serveur est lancé avec: python run_web.py")
    print("Appuyez sur Entrée pour continuer...")
    input()
    
    test_api()
"""
Script de lancement pour l'interface web RAG Assistant
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import uvicorn

def main():
    """Lance le serveur web avec l'interface RAG"""
    
    print("Lancement de VectorBT RAG Assistant Web Interface")
    print("=" * 50)
    
    # Vérifier que les fichiers nécessaires existent
    required_files = [
        "static/index.html",
        "static/style.css", 
        "static/app.js",
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Fichiers manquants:")
        for f in missing_files:
            print(f"   - {f}")
        sys.exit(1)
    
    print("✅ Tous les fichiers requis sont présents")
    print("\n📡 Démarrage du serveur...")
    print("🌐 Interface web disponible sur: http://localhost:8000")
    print("📚 API docs disponible sur: http://localhost:8000/docs")
    print("\n💡 Appuyez sur Ctrl+C pour arrêter le serveur")
    print("=" * 50)
    
    try:
        uvicorn.run(
            "src.api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            reload_dirs=[project_root]
        )
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt du serveur. Au revoir!")

if __name__ == "__main__":
    main()







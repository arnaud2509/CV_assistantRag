# api_rag.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import requests
from dotenv import load_dotenv

# --- Imports LangChain ---
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# --------------------------------------------------------------------------
# ------------------- 1. CONFIGURATION ET LOGIQUE RAG ----------------------
# --------------------------------------------------------------------------

# Charge les variables d'environnement (nécessaire en local, Render les injectera directement)
load_dotenv() 

# Clés
API_KEY = os.environ.get("INFOMANIAK_API_KEY") 
PRODUCT_ID = os.environ.get("INFOMANIAK_PRODUCT_ID")
MODEL = os.environ.get("INFOMANIAK_EMBEDDING_MODEL", "mini_lm_l12_v2")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Chemin de la BDD vectorielle persistante
CHROMA_PERSIST_DIRECTORY = "chroma_db"
# Fichier de données source
DATA_FILE = "cv_rag.json"


# --- Définition de la classe InfomaniakEmbeddings (tirée de rag_app.py) ---
class InfomaniakEmbeddings(Embeddings):
    """Classe d'embeddings personnalisée pour l'API Infomaniak."""
    
    def __init__(self, api_key: str, product_id: str, model: str):
        self.api_key = api_key
        self.product_id = product_id
        self.model = model
        self.base_url = "https://api.infomaniak.com/2/ai/embedding/create"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> list[float]:
        """Appel réel à l'API Infomaniak."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "ProductId": self.product_id
        }
        payload = {
            "model": self.model,
            "input": text
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP
            data = response.json()
            # On s'attend à ce que l'embedding soit une liste de floats
            if data and data.get("data") and data["data"].get("embedding"):
                return data["data"]["embedding"]
            raise ValueError("Réponse API Infomaniak invalide ou incomplète.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Erreur lors de l'appel à l'API Infomaniak: {e}")


def load_documents_and_setup_rag():
    """Charge les documents, configure ChromaDB et le chaîne QA."""
    
    # Vérification des clés essentielles
    if not GEMINI_API_KEY or not API_KEY:
        raise ValueError("Les clés API (GEMINI_API_KEY ou INFOMANIAK_API_KEY) n'ont pas été configurées. Vérifiez les variables d'environnement.")

    # 1. Configuration des Embeddings
    embedding_function = InfomaniakEmbeddings(
        api_key=API_KEY, 
        product_id=PRODUCT_ID, 
        model=MODEL
    )
    
    # 2. Chargement ou création de la base de données vectorielle (ChromaDB)
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        # Charge la base de données existante (recommandé pour la production)
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=embedding_function
        )
    else:
        # Crée la base de données (si elle n'existe pas - lourd au démarrage du serveur)
        print(f"ATTENTION : Le dossier '{CHROMA_PERSIST_DIRECTORY}' n'existe pas. Création des embeddings...")
        
        # Charger les données du fichier JSON (supposé être une liste de dict {page_content, metadata})
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            documents = [Document(page_content=item['page_content'], metadata=item['metadata']) for item in data]
        
        vectorstore = Chroma.from_documents(
            documents=documents, 
            embedding=embedding_function, 
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        vectorstore.persist() # Rend les données persistantes
        print("Embeddings créés et persistés. Il est conseillé de commiter le dossier chroma_db.")


    # 3. Configuration du LLM (Gemini)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        api_key=GEMINI_API_KEY 
    )

    # 4. Création de la chaîne RAG
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    
    return qa

# Initialisation globale de la chaîne QA
try:
    qa_chain = load_documents_and_setup_rag()
except Exception as e:
    # Si le RAG ne peut pas s'initialiser (ex: clé manquante), on lève une erreur critique
    print(f"ERREUR CRITIQUE D'INITIALISATION RAG: {e}")
    qa_chain = None # Laisse qa_chain à None ou arrête l'application si non récupérable.

# --------------------------------------------------------------------------
# ---------------------------- 2. API FASTAPI ------------------------------
# --------------------------------------------------------------------------

app = FastAPI(title="CV RAG API")

# Configuration CORS : Permet à votre frontend (GitHub Pages) d'accéder à cette API.
# Mettez l'URL exacte de votre GitHub Page pour plus de sécurité en production : 
# ex: ["https://arnaud2509.github.io"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporaire: Autorise toutes les origines
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle de données pour la requête POST
class Query(BaseModel):
    question: str

@app.get("/")
def read_root():
    """Point de terminaison de vérification simple."""
    status = "OK" if qa_chain else "ERREUR: RAG non initialisé. Vérifiez les logs/clés API."
    return {"status": status}


@app.post("/ask")
async def ask_rag(query: Query):
    """Point de terminaison pour interroger le RAG (Récupération et Génération Augmentée)."""
    if not qa_chain:
        raise HTTPException(status_code=503, detail="Service RAG indisponible. L'initialisation a échoué.")
        
    try:
        # Exécute la chaîne RAG avec la question reçue
        answer = qa_chain.run(query.question)
        
        # Retourne la réponse dans un format JSON simple
        return {"answer": answer}
        
    except Exception as e:
        # Capture les erreurs lors de l'exécution du RAG
        print(f"Erreur lors de la chaîne QA : {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur interne lors de la génération de la réponse RAG. Détails: {e}"
        )


if __name__ == "__main__":
    import uvicorn
    # Assurez-vous d'avoir vos variables d'environnement dans un .env
    # Lancement : uvicorn api_rag:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
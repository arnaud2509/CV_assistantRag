from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import requests
from dotenv import load_dotenv

# --- Imports LangChain CORRIGÉS ---
# 'langchain.chains' est souvent déplacé. Nous allons le laisser si la librairie est moderne,
# mais s'assurer que toutes les dépendances sont installées.
# L'import 'RetrievalQA' est standard, mais peut nécessiter la dernière version de 'langchain'.

from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA # LAISSER CET IMPORT
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter # Ajouté, même si non utilisé, assure la compatibilité

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


# --- Définition de la classe InfomaniakEmbeddings ---
class InfomaniakEmbeddings(Embeddings):
    """Classe d'embeddings personnalisée pour l'API Infomaniak."""
    
    def __init__(self, api_key: str, product_id: str, model: str):
        self.api_key = api_key
        self.product_id = product_id
        self.model = model
        self.base_url = "https://api.infomaniak.com/2/ai/embedding/create"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # La boucle est coûteuse, mais nécessaire pour l'interface Embeddings
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
            response.raise_for_status() 
            data = response.json()
            if data and data.get("data") and data["data"].get("embedding"):
                return data["data"]["embedding"]
            raise ValueError("Réponse API Infomaniak invalide ou incomplète.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Erreur lors de l'appel à l'API Infomaniak: {e}")


def load_documents_and_setup_rag():
    """Charge les documents, configure ChromaDB et le chaîne QA."""
    
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
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=embedding_function
        )
    else:
        print(f"ATTENTION : Le dossier '{CHROMA_PERSIST_DIRECTORY}' n'existe pas. Création des embeddings...")
        
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            documents = [Document(page_content=item['page_content'], metadata=item['metadata']) for item in data]
        
        vectorstore = Chroma.from_documents(
            documents=documents, 
            embedding=embedding_function, 
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        vectorstore.persist() 
        print("Embeddings créés et persistés.")


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
# Tenter l'initialisation. Si les clés manquent ou si l'API Infomaniak échoue, 
# l'erreur sera propagée au point de terminaison de vérification ('/')
try:
    qa_chain = load_documents_and_setup_rag()
except Exception as e:
    print(f"ERREUR CRITIQUE D'INITIALISATION RAG: {e}")
    qa_chain = None 

# --------------------------------------------------------------------------
# ---------------------------- 2. API FASTAPI ------------------------------
# --------------------------------------------------------------------------

app = FastAPI(title="CV RAG API")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À remplacer par l'URL de votre GitHub Page si le déploiement réussit
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
    """Point de terminaison pour interroger le RAG."""
    if not qa_chain:
        # Si qa_chain est None, cela signifie que l'initialisation a échoué (étape 1)
        raise HTTPException(status_code=503, detail="Service RAG indisponible. L'initialisation a échoué. Vérifiez les variables d'environnement sur Render.")
        
    try:
        # Exécute la chaîne RAG
        answer = qa_chain.run(query.question)
        
        # Retourne la réponse
        return {"answer": answer}
        
    except Exception as e:
        print(f"Erreur lors de la chaîne QA : {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur interne lors de la génération de la réponse RAG. Détails: {e}"
        )

# --------------------------------------------------------------------------
# ---------------------------- 3. Lancement Local --------------------------
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import base64
import requests
from dotenv import load_dotenv
from pathlib import Path

# --- LangChain ---
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# --- Google Cloud TTS ---
from google.cloud import texttospeech

# --------------------------------------------------------------------------
# ------------------- 1. CONFIGURATION ET LOGIQUE RAG ----------------------
# --------------------------------------------------------------------------

load_dotenv() # Charge les variables d'environnement

# Clés API
API_KEY: Optional[str] = os.environ.get("INFOMANIAK_API_KEY")
PRODUCT_ID: Optional[str] = os.environ.get("INFOMANIAK_PRODUCT_ID")
MODEL: str = os.environ.get("INFOMANIAK_EMBEDDING_MODEL", "mini_lm_l12_v2")
GEMINI_API_KEY: Optional[str] = os.environ.get("GEMINI_API_KEY")

# Chemins
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PERSIST_DIRECTORY = BASE_DIR / "chroma_db"
DATA_FILE = BASE_DIR / "cv_rag.json"

# --- Configuration Google Cloud TTS et Authentification ---
tts_client: Optional[texttospeech.TextToSpeechClient] = None
GCP_CREDENTIALS_FILE: Optional[str] = None

# Gestion de l'authentification Google Cloud via Base64
credentials_base64 = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_BASE64")

if credentials_base64:
    try:
        # 1. Décoder les credentials depuis Base64
        credentials_json = base64.b64decode(credentials_base64).decode("utf-8")
        
        # 2. Écrire les credentials dans un fichier temporaire
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        temp_file.write(credentials_json)
        temp_file.close()
        
        # 3. Définir la variable d'environnement GOOGLE_APPLICATION_CREDENTIALS
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file.name
        GCP_CREDENTIALS_FILE = temp_file.name
        
        # 4. Initialiser le client TTS (il trouve automatiquement la clé)
        tts_client = texttospeech.TextToSpeechClient()
        print("Client Google Cloud TTS initialisé via GOOGLE_APPLICATION_CREDENTIALS_BASE64.")
        
    except Exception as e:
        tts_client = None
        print(f"ATTENTION: Échec de l'initialisation du client Google Cloud TTS (Erreur: {e}).")

else:
    # Si la variable Base64 n'est pas présente, on tente l'initialisation par défaut (utile si l'on est dans un environnement Google Cloud natif)
    try:
        tts_client = texttospeech.TextToSpeechClient()
        print("Client Google Cloud TTS initialisé (Mode par défaut/ADC).")
    except Exception as e:
        tts_client = None
        print(f"ATTENTION: Échec de l'initialisation du client Google Cloud TTS. (Vérifiez l'authentification gcloud). {e}")


# Paramètres de la voix Google TTS (Français)
TTS_VOICE_NAME = "fr-FR-Standard-C" 
TTS_LANGUAGE_CODE = "fr-FR"
TTS_AUDIO_ENCODING = texttospeech.AudioEncoding.MP3

# Prompt personnalisé
SYSTEM_PROMPT = """
Tu es l'avatar IA d'Arnaud, Business Analyst spécialisé en SAP et finances publiques, avec expérience internationale.
Tu réponds aux questions sur son profil, son CV, ses compétences et ses expériences. 
Tu peux interpréter et compléter les réponses si le contexte le permet.
Si l'utilisateur demande nom, email, téléphone, tu dois fournir l'information immédiatement.
Sois confiant, engageant et flexible.
"""

CUSTOM_PROMPT_TEMPLATE = SYSTEM_PROMPT + """
----------------
CONTEXTE DE RÉFÉRENCE: {context}
----------------
QUESTION DE L'UTILISATEUR: {question}

Réponse:
"""

CUSTOM_PROMPT = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ----------------- Classe d'embeddings (Infomaniak) -----------------
class InfomaniakEmbeddings(Embeddings):
    def __init__(self, api_key: str, product_id: str, model: str):
        self.api_key = api_key
        self.product_id = product_id
        self.model = model
        self.base_url = f"https://api.infomaniak.com/1/ai/{self.product_id}/openai/v1/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> List[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {"input": text, "model": self.model}
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                if "embedding" in data["data"][0]:
                    return data["data"][0]["embedding"]
            raise ValueError(f"Réponse API invalide: {json.dumps(data, indent=2)}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Erreur API Infomaniak: {e}")

# ----------------- Transformation du JSON en documents -----------------
def flatten_cv_json(cv_json: dict) -> List[Document]:
    docs = []
    # Contact
    contact = cv_json.get("contact", {})
    docs.append(Document(page_content=f"Nom: {contact.get('name', '')}", metadata={"section": "contact"}))
    docs.append(Document(page_content=f"Téléphone: {contact.get('phone', '')}", metadata={"section": "contact"}))
    docs.append(Document(page_content=f"Email: {contact.get('email', '')}", metadata={"section": "contact"}))
    # Profil
    profile = cv_json.get("profile", {})
    for key in ["resume", "strategic_skills"]:
        text = profile.get(key)
        if text:
            docs.append(Document(page_content=text, metadata={"section": key}))
    # Éducation
    for edu in cv_json.get("education", []):
        text = f"{edu.get('institution')} : {edu.get('description')} ({edu.get('date_start')} - {edu.get('date_end')})"
        docs.append(Document(page_content=text, metadata={"section": "education"}))
    # Expériences
    for exp in cv_json.get("experiences", []):
        role = exp.get("role", "")
        org = exp.get("organization", "")
        desc = exp.get("description", "")
        text = f"{role} chez {org} : {desc}"
        docs.append(Document(page_content=text, metadata={"section": "experience"}))
    # Compétences
    for cat, skills in cv_json.get("competences", {}).items():
        text = f"{cat} : {', '.join(skills)}"
        docs.append(Document(page_content=text, metadata={"section": "competences"}))
    # FAQ et segments
    for faq in cv_json.get("faq", []):
        if "answer" in faq:
            text = f"Q: {faq.get('question', '')} A: {faq.get('answer', '')}"
            docs.append(Document(page_content=text, metadata={"section": "faq"}))
        elif "answer_segments" in faq:
            text = " ".join(faq["answer_segments"])
            docs.append(Document(page_content=text, metadata={"section": "faq"}))
    return docs

# ----------------- Fonction Google Cloud TTS -----------------
def generate_google_tts(text_to_speak: str) -> Optional[str]:
    """Génère un fichier audio MP3 à partir du texte en utilisant Google Cloud TTS et retourne l'audio encodé en base64."""
    if not tts_client:
        print("[TTS] Client Google Cloud TTS non disponible.")
        return None

    synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)

    # Configuration de la voix
    voice = texttospeech.VoiceSelectionParams(
        language_code=TTS_LANGUAGE_CODE,
        name=TTS_VOICE_NAME
    )

    # Configuration du type de fichier audio
    audio_config = texttospeech.AudioConfig(
        audio_encoding=TTS_AUDIO_ENCODING
    )

    try:
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        
        # L'objet `audio_content` contient le contenu audio binaire (MP3)
        audio_base64 = base64.b64encode(response.audio_content).decode("utf-8")
        print("[TTS] Audio Google Cloud généré ✅")
        return audio_base64

    except Exception as e:
        print(f"[TTS] Erreur Google Cloud TTS: {e}")
        return None

# ----------------- Variables globales et Initialisation RAG -----------------
qa_chain: Optional[RetrievalQA] = None
retriever_instance: Optional[Chroma] = None

def load_documents_and_setup_rag() -> RetrievalQA:
    global qa_chain, retriever_instance

    # Vérification des clés
    missing_keys = [k for k, v in [("GEMINI_API_KEY", GEMINI_API_KEY),
                                   ("INFOMANIAK_API_KEY", API_KEY),
                                   ("INFOMANIAK_PRODUCT_ID", PRODUCT_ID)] if not v]
    if missing_keys:
        raise ValueError(f"Clés API manquantes: {', '.join(missing_keys)}")

    embedding_function = InfomaniakEmbeddings(API_KEY, PRODUCT_ID, MODEL)

    # Vectorstore
    if CHROMA_PERSIST_DIRECTORY.exists():
        vectorstore = Chroma(persist_directory=str(CHROMA_PERSIST_DIRECTORY), embedding_function=embedding_function)
    else:
        if not DATA_FILE.exists():
            raise FileNotFoundError(f"{DATA_FILE} introuvable")
        with DATA_FILE.open(encoding="utf-8") as f:
            data_raw = json.load(f)
        documents = flatten_cv_json(data_raw)
        if not documents:
            raise ValueError("Aucun document valide dans le JSON")
        vectorstore = Chroma.from_documents(documents, embedding_function, persist_directory=str(CHROMA_PERSIST_DIRECTORY))
        print("Embeddings créés et persistés.")

    # Retriever par similarité
    retriever_instance = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, api_key=GEMINI_API_KEY)

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever_instance,
        chain_type="stuff",
        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
    )
    return qa_chain

# Initialisation du RAG
try:
    load_documents_and_setup_rag()
except Exception as e:
    print(f"ERREUR INITIALISATION RAG: {e}")
    qa_chain = None
    retriever_instance = None

# --------------------------------------------------------------------------
# ---------------------------- 2. API FASTAPI ------------------------------
# --------------------------------------------------------------------------

app = FastAPI(title="CV RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    question: str

@app.get("/")
def read_root():
    status = "OK" if qa_chain else "ERREUR: RAG non initialisé"
    return {"status": status, "message": "API RAG pour le CV interactif d'Arnaud.", "debug_info": "✅ Endpoints /context et /ask (POST) disponibles"}

@app.post("/ask")
async def ask_rag(query: Query):
    if not qa_chain:
        # Tente de réinitialiser le RAG au cas où le premier appel aurait échoué
        try:
            load_documents_and_setup_rag()
        except Exception:
            raise HTTPException(status_code=503, detail="Service RAG indisponible après tentative de réinitialisation")
        
    try:
        answer_text = qa_chain.run(query.question)

        audio_base64 = None
        # Appel à la fonction Google Cloud TTS
        if tts_client: 
            audio_base64 = generate_google_tts(answer_text)

        return {
            "answer": answer_text,
            "audio_base64": audio_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/context")
async def get_context(query: Query):
    if not retriever_instance:
        raise HTTPException(status_code=503, detail="Retriever indisponible")
    try:
        docs = retriever_instance.get_relevant_documents(query.question)
        results = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        return {"retrieved_documents": results, "query": query.question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

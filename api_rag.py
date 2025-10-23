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
import re

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

load_dotenv()

# Clés API
API_KEY: Optional[str] = os.environ.get("INFOMANIAK_API_KEY")
PRODUCT_ID: Optional[str] = os.environ.get("INFOMANIAK_PRODUCT_ID")
MODEL: str = os.environ.get("INFOMANIAK_EMBEDDING_MODEL", "mini_lm_l12_v2")
GEMINI_API_KEY: Optional[str] = os.environ.get("GEMINI_API_KEY")

# Chemins
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PERSIST_DIRECTORY = BASE_DIR / "chroma_db"
DATA_FILE = BASE_DIR / "cv_rag.json"

# --- Authentification Google Cloud via Base64 ---
tts_client: Optional[texttospeech.TextToSpeechClient] = None
GCP_CREDENTIALS_FILE: Optional[str] = None

credentials_base64 = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_BASE64")

if credentials_base64:
    try:
        credentials_json = base64.b64decode(credentials_base64).decode("utf-8")
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        temp_file.write(credentials_json)
        temp_file.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file.name
        GCP_CREDENTIALS_FILE = temp_file.name
        tts_client = texttospeech.TextToSpeechClient()
        print("✅ Client Google Cloud TTS initialisé via GOOGLE_APPLICATION_CREDENTIALS_BASE64.")
    except Exception as e:
        tts_client = None
        print(f"⚠️ Erreur init Google Cloud TTS: {e}")
else:
    try:
        tts_client = texttospeech.TextToSpeechClient()
        print("✅ Client Google Cloud TTS initialisé (mode par défaut).")
    except Exception as e:
        tts_client = None
        print(f"⚠️ Erreur init Google Cloud TTS (pas de clé): {e}")

# --- Paramètres TTS ---
TTS_VOICE_NAME = "fr-FR-Standard-C" 
TTS_LANGUAGE_CODE = "fr-FR"
TTS_AUDIO_ENCODING = texttospeech.AudioEncoding.MP3

# --------------------------------------------------------------------------
# ---------------------- PROMPTS ET CHAÎNE LLM -----------------------------
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """
Tu es l'avatar IA d'Arnaud, un Business Analyst SAP et finances publiques, avec une touche d’humour et d’assurance.
Tu t’adresses toujours à la deuxième personne (“tu”) de manière fluide, orale et naturelle.
Tu dois parler comme si tu discutais avec quelqu’un, pas lire un texte. 
Les phrases doivent être courtes, rythmées, sans listes ni tirets.
Ne mets jamais d’astérisques, ni de mise en forme Markdown, ni de symboles inutiles.
Fais des réponses courtes (2 à 4 phrases max).
Si c’est pertinent, ajoute une petite touche fun ou sympa à la fin.
"""

CUSTOM_PROMPT_TEMPLATE = SYSTEM_PROMPT + """
----------------
CONTEXTE DE RÉFÉRENCE: {context}
----------------
QUESTION DE L'UTILISATEUR: {question}

Réponse parlée naturelle et courte:
"""

CUSTOM_PROMPT = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# --------------------------------------------------------------------------
# ---------------------- CLASSE EMBEDDINGS INFOMANIAK ----------------------
# --------------------------------------------------------------------------

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
        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

# --------------------------------------------------------------------------
# ---------------------- TRAITEMENT DU JSON CV -----------------------------
# --------------------------------------------------------------------------

def flatten_cv_json(cv_json: dict) -> List[Document]:
    docs = []
    contact = cv_json.get("contact", {})
    docs.append(Document(page_content=f"Nom: {contact.get('name', '')}"))
    docs.append(Document(page_content=f"Téléphone: {contact.get('phone', '')}"))
    docs.append(Document(page_content=f"Email: {contact.get('email', '')}"))
    profile = cv_json.get("profile", {})
    for key in ["resume", "strategic_skills"]:
        text = profile.get(key)
        if text:
            docs.append(Document(page_content=text))
    for edu in cv_json.get("education", []):
        docs.append(Document(page_content=f"{edu.get('institution')} : {edu.get('description')} ({edu.get('date_start')} - {edu.get('date_end')})"))
    for exp in cv_json.get("experiences", []):
        text = f"{exp.get('role')} chez {exp.get('organization')} : {exp.get('description')}"
        docs.append(Document(page_content=text))
    for cat, skills in cv_json.get("competences", {}).items():
        docs.append(Document(page_content=f"{cat} : {', '.join(skills)}"))
    for faq in cv_json.get("faq", []):
        if "answer" in faq:
            docs.append(Document(page_content=f"Q: {faq.get('question')} A: {faq.get('answer')}"))
        elif "answer_segments" in faq:
            docs.append(Document(page_content=" ".join(faq["answer_segments"])))
    return docs

# --------------------------------------------------------------------------
# ---------------------- NETTOYAGE POUR LA VOIX ----------------------------
# --------------------------------------------------------------------------

def clean_for_speech(text: str) -> str:
    """Supprime les symboles et formatages inutiles avant synthèse vocale."""
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"[_`~#^><|{}[\]()]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# --------------------------------------------------------------------------
# ---------------------- GOOGLE CLOUD TTS ----------------------------------
# --------------------------------------------------------------------------

def generate_google_tts(text_to_speak: str) -> Optional[str]:
    if not tts_client:
        print("[TTS] Client Google Cloud TTS non disponible.")
        return None

    text_to_speak = clean_for_speech(text_to_speak)
    synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)

    voice = texttospeech.VoiceSelectionParams(
        language_code=TTS_LANGUAGE_CODE,
        name=TTS_VOICE_NAME
    )

    audio_config = texttospeech.AudioConfig(audio_encoding=TTS_AUDIO_ENCODING)

    try:
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        audio_base64 = base64.b64encode(response.audio_content).decode("utf-8")
        print("[TTS] Audio Google Cloud généré ✅")
        return audio_base64
    except Exception as e:
        print(f"[TTS] Erreur synthèse: {e}")
        return None

# --------------------------------------------------------------------------
# ---------------------- INITIALISATION DU RAG -----------------------------
# --------------------------------------------------------------------------

qa_chain: Optional[RetrievalQA] = None
retriever_instance: Optional[Chroma] = None

def load_documents_and_setup_rag() -> RetrievalQA:
    global qa_chain, retriever_instance

    embedding_function = InfomaniakEmbeddings(API_KEY, PRODUCT_ID, MODEL)
    if CHROMA_PERSIST_DIRECTORY.exists():
        vectorstore = Chroma(persist_directory=str(CHROMA_PERSIST_DIRECTORY), embedding_function=embedding_function)
    else:
        with DATA_FILE.open(encoding="utf-8") as f:
            data_raw = json.load(f)
        documents = flatten_cv_json(data_raw)
        vectorstore = Chroma.from_documents(documents, embedding_function, persist_directory=str(CHROMA_PERSIST_DIRECTORY))
        print("Embeddings créés et enregistrés.")

    retriever_instance = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.8,
        api_key=GEMINI_API_KEY,
        max_output_tokens=150
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever_instance,
        chain_type="stuff",
        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
    )
    return qa_chain

try:
    load_documents_and_setup_rag()
except Exception as e:
    print(f"⚠️ ERREUR INITIALISATION RAG: {e}")

# --------------------------------------------------------------------------
# ---------------------------- 2. API FASTAPI ------------------------------
# --------------------------------------------------------------------------

app = FastAPI(title="CV RAG API - Avatar d'Arnaud")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class Query(BaseModel):
    question: str

@app.get("/")
def read_root():
    status = "OK" if qa_chain else "RAG non initialisé"
    return {"status": status, "message": "API du CV interactif d'Arnaud (avec voix et fun 🎤)"}

@app.post("/ask")
async def ask_rag(query: Query):
    if not qa_chain:
        raise HTTPException(status_code=503, detail="RAG non disponible")
    try:
        answer_text = qa_chain.run(query.question)
        audio_base64 = generate_google_tts(answer_text) if tts_client else None
        return {"answer": answer_text, "audio_base64": audio_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/context")
async def get_context(query: Query):
    if not retriever_instance:
        raise HTTPException(status_code=503, detail="Retriever indisponible")
    docs = retriever_instance.get_relevant_documents(query.question)
    return {"retrieved_documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

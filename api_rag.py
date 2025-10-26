# cv_rag_api.py
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
from dotenv import load_dotenv
from pathlib import Path
import re
import shutil
import time
import logging

# LangChain
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------- Config & Logging -----------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cv_rag")

# Clés API
API_KEY: Optional[str] = os.environ.get("INFOMANIAK_API_KEY")
PRODUCT_ID: Optional[str] = os.environ.get("INFOMANIAK_PRODUCT_ID")
MODEL: str = os.environ.get("INFOMANIAK_EMBEDDING_MODEL", "mini_lm_l12_v2")
GEMINI_API_KEY: Optional[str] = os.environ.get("GEMINI_API_KEY")

# Paths
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PERSIST_DIRECTORY = BASE_DIR / "chroma_db"
DATA_FILE = BASE_DIR / "cv_rag.json"

# ----------------- Prompt (séparé style / tâche) -----------------
SYSTEM_STYLE = "Tu es l'avatar IA d'Arnaud, Business Analyst en SAP et finances publiques. Ton ton : concis, clair et naturel à l'oral."
TASK_PROMPT_TEMPLATE = """
Contexte (extraits pertinents) :
{context}

Question : {question}

Réponds en maximum 2 phrases, sans astérisques ni tirets superflus. Sois engageant et naturel.
"""
CUSTOM_PROMPT = PromptTemplate(template=SYSTEM_STYLE + "\n" + TASK_PROMPT_TEMPLATE,
                               input_variables=["context", "question"])

# ----------------- Embeddings wrapper (Infomaniak) -----------------
class InfomaniakEmbeddings(Embeddings):
    """
    Wrapper simple qui effectue retries/backoff et retourne embeddings batch.
    """
    def __init__(self, api_key: str, product_id: str, model: str, max_retries: int = 3, backoff_factor: float = 0.5):
        if not api_key or not product_id:
            raise ValueError("Infomaniak API key et product id requis.")
        self.api_key = api_key
        self.product_id = product_id
        self.model = model
        self.max_retries = int(max_retries)
        self.backoff_factor = backoff_factor
        self.base_url = f"https://api.infomaniak.com/1/ai/{self.product_id}/openai/v1/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # batch-friendly: on itère mais en gérant retries pour chaque texte
        return [self._embed_with_backoff(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_with_backoff(text)

    def _embed_with_backoff(self, text: str) -> List[float]:
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {"input": text, "model": self.model}
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(self.base_url, headers=headers, json=payload, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                # structure attendue: {"data":[{"embedding": [...]}], ...}
                if isinstance(data, dict) and "data" in data and data["data"]:
                    emb = data["data"][0].get("embedding")
                    if isinstance(emb, list):
                        return emb
                raise RuntimeError(f"Format réponse inattendu: {data}")
            except Exception as e:
                sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                logger.warning(f"[Infomaniak] tentative {attempt}/{self.max_retries} échouée: {e} — sleep {sleep_time}s")
                time.sleep(sleep_time)
        # fallback : vecteur neutre (dimension approximative) pour éviter crash ; préférable: rejeter l'indexation
        logger.error("Toutes les tentatives embeddings ont échoué — retour vecteur neutre")
        return [0.0] * 384  # adapter selon la dimension attendue par ton modèle

# ----------------- Flatten CV JSON -----------------
def flatten_cv_json(cv_json: dict) -> List[Document]:
    docs = []
    contact = cv_json.get("contact", {})
    docs.append(Document(page_content=f"Nom: {contact.get('name', '')}", metadata={"section": "contact"}))
    docs.append(Document(page_content=f"Téléphone: {contact.get('phone', '')}", metadata={"section": "contact"}))
    docs.append(Document(page_content=f"Email: {contact.get('email', '')}", metadata={"section": "contact"}))

    profile = cv_json.get("profile", {})
    for key in ["resume", "strategic_skills"]:
        text = profile.get(key)
        if text:
            docs.append(Document(page_content=text, metadata={"section": key}))

    for edu in cv_json.get("education", []):
        text = f"{edu.get('institution')} : {edu.get('description')} ({edu.get('date_start')} - {edu.get('date_end')})"
        docs.append(Document(page_content=text, metadata={"section": "education"}))

    for exp in cv_json.get("experiences", []):
        text = f"{exp.get('role', '')} chez {exp.get('organization', '')} : {exp.get('description', '')}"
        docs.append(Document(page_content=text, metadata={"section": "experience"}))

    for cat, skills in cv_json.get("competences", {}).items():
        text = f"{cat} : {', '.join(skills)}"
        docs.append(Document(page_content=text, metadata={"section": "competences"}))

    # ✅ Ajout langues
    for lang in cv_json.get("languages", []):
        text = f"Langue: {lang.get('langue', '')}, niveau: {lang.get('niveau', '')}"
        docs.append(Document(page_content=text, metadata={"section": "languages"}))

    # ✅ Ajout vision
    vision = cv_json.get("vision", {})
    if "objectives" in vision:
        docs.append(Document(page_content=vision["objectives"], metadata={"section": "vision"}))

    # ✅ Correction FAQ
    for faq in cv_json.get("faq", []):
        if "content" in faq:
            text = f"{faq.get('topic', '')} : {faq.get('content', '')}"
            docs.append(Document(page_content=text, metadata={"section": "faq"}))

    return docs


# ----------------- Nettoyage pour TTS / sortie -----------------
def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"[\*\-_]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ----------------- RAG initialisation + helpers -----------------
qa_chain: Optional[RetrievalQA] = None
retriever_instance = None

def _chroma_persist_valid(persist_dir: Path) -> bool:
    """
    Vérifie la présence de fichiers typiques Chroma (sqlite, parquets...). Adaptable selon la build.
    """
    if not persist_dir.exists():
        return False
    # heuristique simple : présence d'un fichier .sqlite ou d'un 'index' folder
    if any(persist_dir.glob("*.sqlite")) or any(persist_dir.glob("*.parquet")) or (persist_dir / "index").exists():
        return True
    # fallback : si dossier non vide
    return any(persist_dir.iterdir())

def build_and_persist_index(force_reindex: bool = False) -> Chroma:
    """
    Crée le vectorstore (avec chunking) et le persiste. Utiliser force_reindex=True pour régénérer.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} introuvable")
    with DATA_FILE.open(encoding="utf-8") as f:
        data_raw = json.load(f)

    # 1) flatten -> documents
    base_docs = flatten_cv_json(data_raw)
    if not base_docs:
        raise ValueError("Aucun document valide dans le JSON")

    # 2) chunking (300 tokens ~ 200-300 words) ; overlap pour contexte
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(base_docs)

    # 3) prepare embeddings
    emb = InfomaniakEmbeddings(API_KEY, PRODUCT_ID, MODEL)

    # 4) recreate persist dir if forced
    if force_reindex and CHROMA_PERSIST_DIRECTORY.exists():
        logger.info("Suppression du dossier Chroma existant pour reindexation forcée...")
        shutil.rmtree(CHROMA_PERSIST_DIRECTORY)

    CHROMA_PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # 5) build & persist
    logger.info(f"Indexation de {len(split_docs)} chunks dans Chroma...")
    vectorstore = Chroma.from_documents(split_docs, embedding_function=emb, persist_directory=str(CHROMA_PERSIST_DIRECTORY))
    logger.info("Indexation terminée et persistée.")
    return vectorstore

def load_documents_and_setup_rag(force_reindex: bool = False) -> RetrievalQA:
    """
    Initialise qa_chain global (safe checks + valid persist dir).
    """
    global qa_chain, retriever_instance

    # check keys
    missing = [k for k, v in [("GEMINI_API_KEY", GEMINI_API_KEY),
                              ("INFOMANIAK_API_KEY", API_KEY),
                              ("INFOMANIAK_PRODUCT_ID", PRODUCT_ID)] if not v]
    if missing:
        raise ValueError(f"Clés API manquantes: {', '.join(missing)}")

    # create embeddings wrapper
    embedding_function = InfomaniakEmbeddings(API_KEY, PRODUCT_ID, MODEL)

    # decide to load or build
    if not force_reindex and _chroma_persist_valid(CHROMA_PERSIST_DIRECTORY):
        logger.info("Chargement du vectorstore Chroma existant...")
        vectorstore = Chroma(persist_directory=str(CHROMA_PERSIST_DIRECTORY), embedding_function=embedding_function)
    else:
        vectorstore = build_and_persist_index(force_reindex=force_reindex)

    # retriever : k plus large pour CV
    retriever_instance = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, api_key=GEMINI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever_instance, chain_type="stuff",
                                          chain_type_kwargs={"prompt": CUSTOM_PROMPT})
    return qa_chain

# préchargement (non fatal : on loggue)
try:
    load_documents_and_setup_rag()
    logger.info("Initialisation RAG OK")
except Exception as e:
    logger.error(f"ERREUR INITIALISATION RAG: {e}")
    qa_chain = None
    retriever_instance = None

# ----------------- FastAPI -----------------
app = FastAPI(title="CV RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    question: str

@app.get("/")
def read_root():
    status = "OK" if qa_chain else "ERREUR: RAG non initialisé"
    return {"status": status, "message": "API RAG pour le CV interactif d'Arnaud.", "debug_info": "Endpoints: POST /ask, POST /context, POST /reload"}

@app.post("/ask")
async def ask_rag(query: Query):
    global qa_chain
    if not qa_chain:
        try:
            load_documents_and_setup_rag()
        except Exception as e:
            logger.error(f"Impossible d'initialiser le RAG au moment de la requête: {e}")
            raise HTTPException(status_code=503, detail="Service RAG indisponible")
    try:
        answer_text = qa_chain.run(query.question)
        answer_text = clean_text_for_tts(answer_text)
        return {"answer": answer_text}
    except Exception as e:
        logger.exception("Erreur lors du run QA")
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
        logger.exception("Erreur get_context")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
async def reload_index(force: bool = False):
    """
    Forcer la reindexation du CV (utile si cv_rag.json a changé).
    """
    global qa_chain, retriever_instance
    try:
        load_documents_and_setup_rag(force_reindex=True)
        return {"status": "OK", "message": "Réindexation terminée."}
    except Exception as e:
        logger.exception("Erreur reload")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

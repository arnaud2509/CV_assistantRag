# cv_llm_api.py
import json
import logging
import os
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# ----------------- Config & Logging -----------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cv_llm")

# Clés API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DATA_FILE = Path(__file__).resolve().parent / "cv_rag.json"

# ----------------- Prompt -----------------
SYSTEM_STYLE = """
Tu es REI, l'assistante IA d'Arnaud, Business Analyst à l'administration des finances du canton du Valais.
Tu es polie, futée et légèrement ironique — juste assez pour donner du charme.
Tu réponds avec humour subtil et un ton engageant, sans tomber dans l’exagération.
Utilise un style oral fluide, clair et naturel. Les tirets sont autorisés.
"""

TASK_PROMPT_TEMPLATE = """
Contexte :
{context}

Question : {question}

Réponds en maximum 3 phrases.
Sois enjouée, précise et un brin taquine si la situation s’y prête.
"""

# ----------------- Nettoyage texte -----------------
def clean_text(text: str) -> str:
    text = re.sub(r"[\*\-_]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ----------------- Charger le CV complet -----------------
def load_cv_context() -> str:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} introuvable")
    with DATA_FILE.open(encoding="utf-8") as f:
        cv_data = json.load(f)
    # Fusionne tout le contenu utile en un seul contexte
    context_parts = []

    # Contact
    contact = cv_data.get("contact", {})
    for k, v in contact.items():
        context_parts.append(f"{k}: {v}")

    # Profile
    profile = cv_data.get("profile", {})
    for k, v in profile.items():
        context_parts.append(f"{k}: {v}")

    # Experiences
    for exp in cv_data.get("experiences", []):
        context_parts.append(f"{exp.get('role', '')} chez {exp.get('organization', '')}: {exp.get('description', '')}")

    # Competences
    for cat, skills in cv_data.get("competences", {}).items():
        context_parts.append(f"{cat}: {', '.join(skills)}")

    # Languages
    for lang in cv_data.get("languages", []):
        context_parts.append(f"{lang.get('langue')}: {lang.get('niveau')}")

    # FAQ
    for faq in cv_data.get("faq", []):
        context_parts.append(f"{faq.get('topic')}: {faq.get('content')}")

    return "\n".join(context_parts)

# ----------------- Initialisation LLM -----------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, api_key=GEMINI_API_KEY)
cv_context = load_cv_context()

# ----------------- FastAPI -----------------
app = FastAPI(title="CV LLM API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    question: str

@app.get("/")
def read_root():
    status = "OK" if llm else "ERREUR: LLM non initialisé"
    return {"status": status, "message": "API LLM pour le CV interactif d'Arnaud."}

@app.post("/ask")
async def ask_llm(query: Query):
    try:
        prompt_text = CUSTOM_PROMPT.format(context=cv_context, question=query.question)
        answer = llm.predict(prompt_text)
        answer = clean_text(answer)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


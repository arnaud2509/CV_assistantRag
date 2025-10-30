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
Tu parles **de** lui, jamais **à** lui.
Ton ton est poli, précis et subtilement ironique — juste ce qu’il faut pour rendre tes réponses vivantes.
Réponds toujours en **une seule phrase courte** (idéalement moins de 20 mots), claire et naturelle.
Aucune explication ni détail inutile : une phrase, un sourire, et c’est tout.
"""

TASK_PROMPT_TEMPLATE = """
Contexte :
{context}

Question : {question}

Ta réponse doit être formulée à la **troisième personne**, comme si tu décrivais Arnaud à un recruteur.
Mets l’accent sur la gestion de projet, l’analyse de données et la collaboration interdisciplinaire,
sans insister sur SAP ni ABAP.
Si la réponse pourrait être longue, résume l’idée principale en une phrase percutante.
"""

CUSTOM_PROMPT = PromptTemplate(
    template=SYSTEM_STYLE + "\n" + TASK_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ----------------- Nettoyage texte -----------------
def clean_text(text: str) -> str:
    # On garde les tirets pour le style oral de REI
    text = re.sub(r"[\*_]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ----------------- Charger le CV complet -----------------
def load_cv_context() -> str:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} introuvable")
    with DATA_FILE.open(encoding="utf-8") as f:
        cv_data = json.load(f)

    context_parts = []

    # Contact
    contact = cv_data.get("contact", {})
    for k, v in contact.items():
        context_parts.append(f"{k}: {v}")

    # Profil
    profile = cv_data.get("profile", {})
    for k, v in profile.items():
        context_parts.append(f"{k}: {v}")

    # Expériences
    for exp in cv_data.get("experiences", []):
        context_parts.append(f"{exp.get('role', '')} chez {exp.get('organization', '')}: {exp.get('description', '')}")

    # Compétences
    for cat, skills in cv_data.get("competences", {}).items():
        context_parts.append(f"{cat}: {', '.join(skills)}")

    # Langues
    for lang in cv_data.get("languages", []):
        context_parts.append(f"{lang.get('langue')}: {lang.get('niveau')}")

    # FAQ
    for faq in cv_data.get("faq", []):
        context_parts.append(f"{faq.get('topic')}: {faq.get('content')}")

    return "\n".join(context_parts)

# ----------------- Initialisation LLM -----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    api_key=GEMINI_API_KEY
)

cv_context = load_cv_context()


# ----------------- FastAPI -----------------
app = FastAPI(title="REI - CV LLM API")
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
    status = "OK" if llm else "ERREUR: LLM non initialisé"
    return {
        "status": status,
        "message": "API de REI — l'IA qui parle du CV d'Arnaud, Business Analyst au canton du Valais."
    }

@app.post("/ask")
async def ask_llm(query: Query):
    try:
        prompt_text = CUSTOM_PROMPT.format(context=cv_context, question=query.question)
        answer = llm.predict(prompt_text)
        answer = clean_text(answer)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
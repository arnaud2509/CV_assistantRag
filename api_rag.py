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
# 💡 Nouveau style : ton poli, ironique, fun
SYSTEM_STYLE = """
Réponds en **une seule phrase courte** (idéalement moins de 20 mots).
Aucune explication, aucun détail inutile — juste l’essentiel avec ton ton poli et ta petite touche d’ironie.
Si la réponse dépasse une phrase, interrompt-toi et conclus brièvement.
"""

TASK_PROMPT_TEMPLATE = """
Contexte :
{context}

Question : {question}

Réponds en **maximum une phrases complètes**, pas plus.
Sois enjouée, précise et un brin taquine si la situation s’y prête.
Si la réponse risque d’être longue, résume l’idée principale en une phrase claire et naturelle.
Le poste est celui d’un chef de projet IA — mets donc en avant la gestion de projet, l’analyse de données et la collaboration interdisciplinaire, sans insister sur SAP ni ABAP.
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
        "message": "API de REI — l'IA qui parle (avec humour) du CV d'Arnaud, Business Analyst au canton du Valais."
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
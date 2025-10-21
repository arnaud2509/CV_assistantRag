import streamlit as st
import requests
import json
import os
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# --- 1. CHARGEMENT ENVIRONNEMENT ---
# Charge les variables du fichier .env pour les rendre disponibles via os.environ
load_dotenv() 

# ------------------- CONFIG -------------------
# Clés Infomaniak
API_KEY = os.environ.get("INFOMANIAK_API_KEY") 
PRODUCT_ID = os.environ.get("INFOMANIAK_PRODUCT_ID")
MODEL = os.environ.get("INFOMANIAK_EMBEDDING_MODEL", "mini_lm_l12_v2") # Défaut pour la robustesse

# Clé Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- LIGNE CRUCIALE pour l'authentification Gemini (contre l'erreur ADC) ---
if not GEMINI_API_KEY:
    st.error("ERREUR : La clé GEMINI_API_KEY n'a pas été trouvée. Veuillez vérifier votre fichier .env.")
    st.stop()
# L'assignation à os.environ est retirée, la clé sera passée directement au constructeur.

# Vérification des clés Infomaniak
if not API_KEY or not PRODUCT_ID:
    st.warning("Attention: Les clés Infomaniak (API_KEY ou PRODUCT_ID) sont manquantes dans le fichier .env. L'indexation peut échouer.")


# ------------------- Fonction pour récupérer embeddings via Infomaniak -------------------
def get_embedding(text: str):
    """Appelle l'API Infomaniak pour générer un embedding."""
    url = f"https://api.infomaniak.com/1/ai/{PRODUCT_ID}/openai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": MODEL, "input": text} 
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() # Lève une exception pour les codes 4xx/5xx (ex: 401, 404, 422)
    except requests.exceptions.HTTPError as e:
        # ✅ Log détaillé en cas d'erreur HTTP pour diagnostiquer le 422
        if e.response.status_code == 422:
            st.error(f"Erreur 422 (Contenu Irrécupérable) : Le corps de la requête (payload) est rejeté.")
            st.code(f"Payload envoyé : {payload}")
            st.code(f"Réponse complète de l'API : {e.response.text}")
            st.stop()
        else:
            raise e # Rélancer d'autres erreurs HTTP
    
    response_data = response.json()
    
    # Correction KeyError: L'embedding se trouve dans ['data'][0]['embedding']
    if response_data and "data" in response_data and len(response_data["data"]) > 0:
        return response_data["data"][0]["embedding"]
    else:
        raise ValueError("L'API Infomaniak n'a pas renvoyé un embedding valide.")

# ------------------- Classe Embeddings pour LangChain -------------------
class InfomaniakEmbeddings(Embeddings):
    """Classe Wrapper pour intégrer l'API Infomaniak comme fonction d'embedding LangChain."""
    def embed_documents(self, texts):
        return [get_embedding(t) for t in texts]

    def embed_query(self, text):
        return get_embedding(text)

# ------------------- Charger le JSON CV -------------------
try:
    with open("cv_rag.json", "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    st.error("Le fichier 'cv_rag.json' est introuvable. Assurez-vous qu'il est dans le même dossier.")
    st.stop()

# Extraire le texte de chaque section et créer des documents LangChain
texts = [entry["texte"] for entry in data]
docs = [Document(page_content=t) for t in texts]

# ------------------- Créer l'instance embeddings et le VectorStore -------------------
embedding_model = InfomaniakEmbeddings()

try:
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embedding_model, 
        collection_name="cv_ai"
    )
    # L'indexation a réussi si le code arrive ici
    st.sidebar.success("✅ Indexation RAG (Infomaniak) réussie.")
except requests.exceptions.HTTPError:
    # L'erreur 422 est déjà gérée par st.stop()
    pass 
except Exception as e:
    st.error(f"Erreur fatale lors de la création du VectorStore. Détails: {e}")
    st.stop()


# ------------------- Configurer le Retriever + RAG -------------------
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# LLM : Maintenant Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=GEMINI_API_KEY # ✅ CORRECTION FINALE : Passage direct de la clé
)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# ------------------- Interface Streamlit -------------------
st.set_page_config(page_title="CV interactif IA", layout="wide")
st.title("CV interactif IA - RAG avec Infomaniak et Gemini")
st.write("Pose une question à mon avatar IA :")

question = st.text_input("Ta question ici :")

if question:
    with st.spinner("L'avatar réfléchit..."):
        try:
            answer = qa.run(question)
            st.success(answer)
        except Exception as e:
            st.error(f"Une erreur s'est produite lors de la génération de la réponse. Détails: {e}")
            
# --- Ajout d'une section pour démontrer la stratégie ---
st.markdown("---")
st.subheader("💡 Architecture démontrée pour le poste de Chef de Projet IA")
st.markdown("""
- **Souveraineté des Données :** Utilisation d'**Infomaniak Embeddings** (service Suisse/Valais) pour l'indexation du CV.
- **Performance LLM :** Utilisation de **Gemini 2.5 Flash** pour la qualité de la réponse et la rapidité d'exécution.
- **Interopérabilité (API) :** Le RAG est construit de manière modulaire, pouvant être facilement exposé via une API **FastAPI** pour intégration dans n'importe quel système de l'administration cantonale (comme discuté précédemment).
""")
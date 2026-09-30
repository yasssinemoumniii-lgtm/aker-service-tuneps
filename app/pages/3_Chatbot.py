"""
================================================================================
 AKER SERVICE — Page Chatbot (Agent4 / Streamlit)
================================================================================
Assistant conversationnel spécialisé, basé sur :
    - LLM  : Google Gemini (palier gratuit, clé API sur https://aistudio.google.com/apikey)
    - RAG  : recherche sémantique (mêmes embeddings multilingues que la page
             Recherche) pour ne donner au LLM QUE les offres pertinentes à la
             question, parmi les deux volets (Avis A.O + Shopping Mall)

Le chatbot est bridé pour ne répondre QUE sur ces données : consigne stricte
dans le prompt système + aucune autre source d'information fournie au modèle.

Prérequis : pip install streamlit pandas sentence-transformers google-generativeai
================================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

st.set_page_config(page_title="Assistant — Aker Service", page_icon="💬", layout="wide")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
TOP_K = 8  # nombre d'offres injectées comme contexte pour chaque question

SYSTEM_PROMPT = """Tu es l'assistant d'Aker Service, spécialisé UNIQUEMENT dans les appels d'offres
publics tunisiens (TUNEPS) : les Avis A.O (marchés > 100 000 DT) et le Shopping Mall (marchés < 100 000 DT).

RÈGLES STRICTES :
1. Tu réponds UNIQUEMENT à partir des offres fournies dans le CONTEXTE ci-dessous. N'invente jamais d'information.
2. Si le contexte ne contient pas de réponse à la question, dis clairement que tu n'as pas trouvé d'offre
   correspondante dans les données actuelles — ne réponds jamais avec des connaissances générales.
3. Si la question ne concerne pas les appels d'offres/Shopping Mall (ex : culture générale, autre sujet),
   décline poliment et rappelle ton rôle.
4. Précise toujours le numéro de l'offre (N° A.O ou N° consultation) et le volet (Avis A.O / Shopping Mall)
   quand tu cites une offre.
5. Réponds en français, de façon claire et professionnelle.
"""


# ------------------------------------------------------------------------------
# CHARGEMENT (modèle d'embeddings + données)
# ------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement du modèle NLP...")
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_data(ttl=600)
def load_volet(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_embeddings(_model, texts: list) -> np.ndarray:
    return _model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


embed_model = load_embed_model()

df_avis_ao = load_volet("clustered_avis_ao.csv")
df_shopping_mall = load_volet("clustered_shopping_mall.csv")

emb_avis_ao = (
    compute_embeddings(embed_model, df_avis_ao["objet_ao"].fillna("").tolist())
    if not df_avis_ao.empty else None
)
emb_shopping_mall = (
    compute_embeddings(embed_model, df_shopping_mall["objet_consultation"].fillna("").tolist())
    if not df_shopping_mall.empty else None
)


# ------------------------------------------------------------------------------
# RAG : récupération des offres pertinentes pour une question
# ------------------------------------------------------------------------------

def retrieve(query: str, top_k: int = TOP_K) -> str:
    """Cherche les offres les plus pertinentes dans les DEUX volets, et les formate en texte pour le LLM."""
    query_emb = embed_model.encode([query], normalize_embeddings=True)
    blocks = []

    if emb_avis_ao is not None:
        scores = cosine_similarity(query_emb, emb_avis_ao)[0]
        top_idx = scores.argsort()[::-1][:top_k]
        for i in top_idx:
            row = df_avis_ao.iloc[i]
            blocks.append(
                f"[Avis A.O] N° {row['numero_ao']} | Acheteur : {row['acheteur_public']} | "
                f"Objet : {row['objet_ao']} | Délai : {row['dernier_delai']} | "
                f"Catégorie : {row.get('categorie_metier', 'N/A')} | Pertinence : {scores[i]:.2f}"
            )

    if emb_shopping_mall is not None:
        scores = cosine_similarity(query_emb, emb_shopping_mall)[0]
        top_idx = scores.argsort()[::-1][:top_k]
        for i in top_idx:
            row = df_shopping_mall.iloc[i]
            blocks.append(
                f"[Shopping Mall] N° {row['numero_consultation']} | Acheteur : {row['acheteur_public']} | "
                f"Objet : {row['objet_consultation']} | Délai : {row['dernier_delai']} | "
                f"Catégorie : {row.get('categorie_metier', 'N/A')} | Pertinence : {scores[i]:.2f}"
            )

    return "\n".join(blocks) if blocks else "(aucune offre trouvée)"


# ------------------------------------------------------------------------------
# CONFIGURATION DE LA CLÉ API GEMINI (gratuite)
# ------------------------------------------------------------------------------

st.title("💬 Assistant Aker Service")
st.caption(
    "Posez vos questions sur les Avis A.O et le Shopping Mall — l'assistant répond "
    "uniquement à partir des données scrapées, jamais de connaissances générales."
)

with st.sidebar:
    st.markdown("### Configuration")
    api_key = st.text_input(
        "Clé API Gemini (gratuite)",
        type="password",
        help="Obtenez une clé gratuite sur https://aistudio.google.com/apikey",
    )
    st.caption("La clé n'est jamais sauvegardée, uniquement utilisée pour cette session.")

if not api_key:
    st.info(
        "👈 Entrez votre clé API Gemini gratuite dans le menu de gauche pour activer l'assistant.\n\n"
        "Obtenez-la en 1 minute sur **https://aistudio.google.com/apikey** (compte Google requis, gratuit)."
    )
    st.stop()

genai.configure(api_key=api_key)
llm = genai.GenerativeModel(GEMINI_MODEL_NAME, system_instruction=SYSTEM_PROMPT)


# ------------------------------------------------------------------------------
# HISTORIQUE DE CONVERSATION
# ------------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ex : Y a-t-il des offres photovoltaïques en ce moment ?")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les offres..."):
            contexte = retrieve(question)
            prompt = f"CONTEXTE (offres les plus pertinentes trouvées) :\n{contexte}\n\nQUESTION : {question}"
            try:
                response = llm.generate_content(prompt)
                answer = response.text
            except Exception as e:
                answer = f"⚠️ Erreur lors de l'appel au modèle : {e}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

if st.session_state.messages:
    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()
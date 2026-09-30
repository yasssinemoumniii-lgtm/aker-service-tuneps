"""
================================================================================
 AKER SERVICE — Page Recherche (Agent4 / Streamlit)
================================================================================
Recherche sémantique (NLP) par embeddings multilingues : l'utilisateur tape
un mot ou une phrase, on retrouve les offres les plus proches PAR LE SENS
(pas juste par mots exacts), y compris entre français et arabe.

Modèle utilisé : paraphrase-multilingual-MiniLM-L12-v2 (léger, multilingue,
bon compromis qualité/vitesse pour ce volume de données).

Prérequis : pip install streamlit pandas sentence-transformers scikit-learn
(le premier lancement télécharge le modèle, ~470 Mo, une seule fois)
================================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recherche — Aker Service", page_icon="🔍", layout="wide")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# ------------------------------------------------------------------------------
# CHARGEMENT (modèle + données + embeddings, tout mis en cache)
# ------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement du modèle NLP (une seule fois)...")
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_data(ttl=600)
def load_volet(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=3600, show_spinner="Calcul des embeddings (une seule fois)...")
def compute_embeddings(_model, texts: list) -> np.ndarray:
    return _model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


model = load_model()

df_avis_ao = load_volet("clustered_avis_ao.csv")
df_shopping_mall = load_volet("clustered_shopping_mall.csv")

emb_avis_ao = (
    compute_embeddings(model, df_avis_ao["objet_ao"].fillna("").tolist())
    if not df_avis_ao.empty else None
)
emb_shopping_mall = (
    compute_embeddings(model, df_shopping_mall["objet_consultation"].fillna("").tolist())
    if not df_shopping_mall.empty else None
)


# ------------------------------------------------------------------------------
# RECHERCHE PAR SIMILARITÉ
# ------------------------------------------------------------------------------

def search(query: str, df: pd.DataFrame, embeddings: np.ndarray, top_n: int = 10) -> pd.DataFrame:
    if not query.strip() or embeddings is None or df.empty:
        return pd.DataFrame()

    query_emb = model.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(query_emb, embeddings)[0]

    result = df.copy()
    result["score"] = scores
    result = result.sort_values("score", ascending=False).head(top_n)
    return result


# ------------------------------------------------------------------------------
# AFFICHAGE D'UN VOLET
# ------------------------------------------------------------------------------

def render_recherche_volet(titre: str, emoji: str, df: pd.DataFrame, embeddings, objet_col: str, numero_col: str, key_prefix: str):
    st.markdown(f"### {emoji} {titre}")

    if df.empty:
        st.info("Aucune donnée disponible pour ce volet.")
        return

    query = st.text_input(
        "Recherchez par mot ou phrase :",
        placeholder="ex : panneaux solaires, matériel informatique, travaux de construction...",
        key=f"{key_prefix}_query",
    )

    if not query:
        st.caption(f"{len(df)} offres actives dans ce volet — tapez une recherche ci-dessus.")
        return

    results = search(query, df, embeddings, top_n=10)

    if results.empty:
        st.warning("Aucun résultat.")
        return

    st.caption(f"{len(results)} résultats les plus proches, triés par pertinence :")

    for _, row in results.iterrows():
        score_pct = round(row["score"] * 100)
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"**{row[objet_col]}**")
                st.caption(f"🔢 N° {row[numero_col]}  ·  📍 {row['acheteur_public']}  ·  📅 Délai : {row['dernier_delai']}")
                if "categorie_metier" in row:
                    st.caption(f"🏷️ {row['categorie_metier']}")
            with c2:
                st.metric("Pertinence", f"{score_pct}%")


# ------------------------------------------------------------------------------
# PAGE
# ------------------------------------------------------------------------------

st.title("🔍 Recherche")
st.caption(
    "Recherche sémantique (NLP) : tapez un mot ou une phrase, le système comprend "
    "le sens de votre recherche (pas juste les mots exacts), en français comme en arabe."
)
st.divider()

col_ao, col_sm = st.columns(2)

with col_ao:
    render_recherche_volet(
        "Avis A.O", "📁", df_avis_ao, emb_avis_ao, "objet_ao", "numero_ao", key_prefix="ao"
    )

with col_sm:
    render_recherche_volet(
        "Shopping Mall", "🛒", df_shopping_mall, emb_shopping_mall, "objet_consultation", "numero_consultation", key_prefix="sm"
    )
"""
================================================================================
 AKER SERVICE — Page d'accueil (Agent4 / Streamlit)
================================================================================
Point d'entrée de l'application : lancer avec

    streamlit run app.py

Cette page :
    - présente le projet clairement (identité, objectif)
    - affiche des indicateurs clés en un coup d'œil (offres actives, dont
      photovoltaïque, fraîcheur des données)
    - propose une navigation claire vers les 3 autres pages
      (Dashboard, Recherche, Chatbot — dans le dossier pages/)

Prérequis : pip install streamlit pandas
================================================================================
"""

from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------------
# CONFIGURATION DE LA PAGE
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Aker Service — Veille TUNEPS",
    page_icon="☀️",
    layout="wide",
)

DATA_DIR = Path(__file__).resolve().parent / "data"


# ------------------------------------------------------------------------------
# STYLE (CSS léger pour un rendu pro)
# ------------------------------------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0;
        background: linear-gradient(90deg, #F5A623, #1B6E43);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 1.15rem;
        color: #666;
        margin-top: 0.2rem;
        margin-bottom: 1.8rem;
    }
    .kpi-card {
        background-color: #f8f9fa;
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        border: 1px solid #eee;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1B6E43;
    }
    .kpi-label {
        font-size: 0.95rem;
        color: #555;
        margin-top: 0.2rem;
    }
    .nav-card {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 16px;
        padding: 1.6rem;
        height: 100%;
    }
    .nav-card h3 {
        margin-top: 0;
    }
    .footer-note {
        color: #999;
        font-size: 0.85rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# CHARGEMENT DES DONNÉES (pour les indicateurs de la page d'accueil)
# ------------------------------------------------------------------------------

@st.cache_data(ttl=600)
def load_data():
    avis_ao_path = DATA_DIR / "clustered_avis_ao.csv"
    shopping_mall_path = DATA_DIR / "clustered_shopping_mall.csv"

    df_avis_ao = pd.read_csv(avis_ao_path) if avis_ao_path.exists() else pd.DataFrame()
    df_shopping_mall = pd.read_csv(shopping_mall_path) if shopping_mall_path.exists() else pd.DataFrame()

    # Heure de dernière mise à jour = date de modification du fichier le plus récent
    last_update = None
    for p in [avis_ao_path, shopping_mall_path]:
        if p.exists():
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
            if last_update is None or mtime > last_update:
                last_update = mtime

    return df_avis_ao, df_shopping_mall, last_update


df_avis_ao, df_shopping_mall, last_update = load_data()

nb_avis_ao = len(df_avis_ao)
nb_shopping_mall = len(df_shopping_mall)

if "categorie_metier" in df_avis_ao.columns:
    nb_pv_ao = (df_avis_ao["categorie_metier"] == "Photovoltaïque / Énergie solaire").sum()
else:
    nb_pv_ao = 0

if "categorie_metier" in df_shopping_mall.columns:
    nb_pv_sm = (df_shopping_mall["categorie_metier"] == "Photovoltaïque / Énergie solaire").sum()
else:
    nb_pv_sm = 0

nb_pv_total = nb_pv_ao + nb_pv_sm


# ------------------------------------------------------------------------------
# EN-TÊTE
# ------------------------------------------------------------------------------

st.markdown('<p class="main-title">☀️ Aker Service</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Veille intelligente des marchés publics tunisiens (TUNEPS) — '
    'Avis d\'Appels d\'Offres &amp; Shopping Mall, avec un focus particulier sur le photovoltaïque</p>',
    unsafe_allow_html=True,
)

if df_avis_ao.empty and df_shopping_mall.empty:
    st.warning(
        "⚠️ Aucune donnée trouvée dans `app/data/`. Lancez d'abord les agents "
        "1 (scraping), 2 (nettoyage) et 3 (clustering) pour générer "
        "`clustered_avis_ao.csv` et `clustered_shopping_mall.csv`."
    )

st.divider()

# ------------------------------------------------------------------------------
# INDICATEURS CLÉS (KPIs)
# ------------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{nb_avis_ao}</div>
        <div class="kpi-label">Avis A.O actifs<br>(&gt; 100 000 DT)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{nb_shopping_mall}</div>
        <div class="kpi-label">Shopping Mall actifs<br>(&lt; 100 000 DT)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card" style="background-color:#fff8ec;">
        <div class="kpi-value" style="color:#F5A623;">{nb_pv_total}</div>
        <div class="kpi-label">☀️ Offres Photovoltaïque<br>(les deux volets)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    maj_str = last_update.strftime("%d/%m/%Y %H:%M") if last_update else "—"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="font-size:1.4rem;">{maj_str}</div>
        <div class="kpi-label">Dernière mise à jour<br>des données</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ------------------------------------------------------------------------------
# NAVIGATION VERS LES AUTRES PAGES
# ------------------------------------------------------------------------------

st.markdown("### Explorer l'application")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown("""
    <div class="nav-card">
        <h3>📊 Dashboard</h3>
        <p>Vue d'ensemble riche des deux volets : répartition par catégorie métier,
        urgence des délais, acheteurs publics les plus actifs.</p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/1_Dashboard.py", label="Ouvrir le Dashboard", icon="📊")

with nav_col2:
    st.markdown("""
    <div class="nav-card">
        <h3>🔍 Recherche</h3>
        <p>Tapez un mot ou une phrase : retrouvez instantanément les Avis A.O
        et offres Shopping Mall les plus similaires.</p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/2_Recherche.py", label="Ouvrir la Recherche", icon="🔍")

with nav_col3:
    st.markdown("""
    <div class="nav-card">
        <h3>💬 Assistant</h3>
        <p>Posez vos questions sur les appels d'offres et le Shopping Mall
        à un assistant spécialisé (LLM + RAG).</p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/3_Chatbot.py", label="Ouvrir l'Assistant", icon="💬")

st.markdown(
    '<p class="footer-note">Aker Service — Données TUNEPS, actualisées automatiquement par le pipeline de scraping.</p>',
    unsafe_allow_html=True,
)
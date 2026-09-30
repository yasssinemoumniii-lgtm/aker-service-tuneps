"""
================================================================================
 AKER SERVICE — Page Dashboard (Agent4 / Streamlit)
================================================================================
Affiche les deux volets (Avis A.O / Shopping Mall) côte à côte, avec pour
chacun :
    1. Répartition par catégorie métier (avec Photovoltaïque mis en avant)
    2. Offres classées par urgence (délai qui approche)
    3. Top 10 des acheteurs publics les plus actifs
    4. Évolution des publications dans le temps

Prérequis : pip install streamlit pandas plotly
================================================================================
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Dashboard — Aker Service", page_icon="📊", layout="wide")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Couleur dédiée au Photovoltaïque pour qu'il ressorte toujours dans les graphiques
COULEUR_PV = "#F5A623"
COULEUR_DEFAUT = px.colors.qualitative.Set2


# ------------------------------------------------------------------------------
# CHARGEMENT DES DONNÉES
# ------------------------------------------------------------------------------

@st.cache_data(ttl=600)
def load_volet(filename: str, date_col: str = "date_publication") -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if date_col in df.columns:
        df[f"{date_col}_dt"] = pd.to_datetime(df[date_col], format="%d/%m/%Y", errors="coerce")
    return df


df_avis_ao = load_volet("clustered_avis_ao.csv")
df_shopping_mall = load_volet("clustered_shopping_mall.csv")


# ------------------------------------------------------------------------------
# FONCTIONS DE GRAPHIQUES (réutilisées pour les deux volets)
# ------------------------------------------------------------------------------

def couleurs_categories(categories):
    """Attribue une couleur dédiée au Photovoltaïque, et une palette classique aux autres."""
    palette = {}
    autres = [c for c in categories if c != "Photovoltaïque / Énergie solaire"]
    for i, cat in enumerate(autres):
        palette[cat] = COULEUR_DEFAUT[i % len(COULEUR_DEFAUT)]
    if "Photovoltaïque / Énergie solaire" in categories:
        palette["Photovoltaïque / Énergie solaire"] = COULEUR_PV
    return palette


def chart_categories(df: pd.DataFrame):
    counts = df["categorie_metier"].value_counts().reset_index()
    counts.columns = ["categorie", "nombre"]
    palette = couleurs_categories(counts["categorie"].tolist())
    fig = px.pie(
        counts, names="categorie", values="nombre",
        color="categorie", color_discrete_map=palette,
        hole=0.45,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=True, height=380, margin=dict(t=10, b=10, l=10, r=10))
    return fig


def chart_urgence(df: pd.DataFrame):
    def bucket(j):
        if pd.isna(j):
            return "Date inconnue"
        if j < 0:
            return "Expiré"
        if j <= 7:
            return "🔴 Urgent (≤ 7j)"
        if j <= 15:
            return "🟠 Cette quinzaine (8-15j)"
        if j <= 30:
            return "🟡 Ce mois (16-30j)"
        return "🟢 Plus tard (> 30j)"

    order = ["🔴 Urgent (≤ 7j)", "🟠 Cette quinzaine (8-15j)", "🟡 Ce mois (16-30j)", "🟢 Plus tard (> 30j)", "Date inconnue"]
    df = df.copy()
    df["urgence"] = df["jours_restants"].apply(bucket) if "jours_restants" in df.columns else "Date inconnue"
    counts = df["urgence"].value_counts().reindex(order).dropna().reset_index()
    counts.columns = ["urgence", "nombre"]

    colors = {
        "🔴 Urgent (≤ 7j)": "#E74C3C", "🟠 Cette quinzaine (8-15j)": "#F39C12",
        "🟡 Ce mois (16-30j)": "#F1C40F", "🟢 Plus tard (> 30j)": "#27AE60",
        "Date inconnue": "#BDC3C7",
    }
    fig = px.bar(
        counts, x="urgence", y="nombre", color="urgence",
        color_discrete_map=colors, text="nombre",
    )
    fig.update_layout(showlegend=False, height=320, xaxis_title="", yaxis_title="Nombre d'offres",
                       margin=dict(t=10, b=10, l=10, r=10))
    return fig


def chart_top_acheteurs(df: pd.DataFrame, top_n: int = 10):
    counts = df["acheteur_public"].value_counts().head(top_n).sort_values().reset_index()
    counts.columns = ["acheteur", "nombre"]
    fig = px.bar(counts, x="nombre", y="acheteur", orientation="h", text="nombre")
    fig.update_traces(marker_color="#1B6E43")
    fig.update_layout(height=380, xaxis_title="Nombre d'offres", yaxis_title="",
                       margin=dict(t=10, b=10, l=10, r=10))
    return fig


def chart_evolution(df: pd.DataFrame):
    if "date_publication_dt" not in df.columns:
        return None
    counts = df.dropna(subset=["date_publication_dt"]).groupby(
        df["date_publication_dt"].dt.date
    ).size().reset_index(name="nombre")
    counts.columns = ["date", "nombre"]
    fig = px.line(counts, x="date", y="nombre", markers=True)
    fig.update_traces(line_color="#1B6E43")
    fig.update_layout(height=300, xaxis_title="", yaxis_title="Offres publiées",
                       margin=dict(t=10, b=10, l=10, r=10))
    return fig


# ------------------------------------------------------------------------------
# RENDU D'UN VOLET COMPLET
# ------------------------------------------------------------------------------

def render_volet(df: pd.DataFrame, titre: str, emoji: str):
    st.markdown(f"## {emoji} {titre}")

    if df.empty:
        st.info("Aucune donnée disponible pour ce volet.")
        return

    nb_total = len(df)
    nb_pv = (df["categorie_metier"] == "Photovoltaïque / Énergie solaire").sum() if "categorie_metier" in df.columns else 0
    nb_urgent = (df["jours_restants"] <= 7).sum() if "jours_restants" in df.columns else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total actives", nb_total)
    c2.metric("☀️ Photovoltaïque", nb_pv)
    c3.metric("🔴 Urgentes (≤7j)", nb_urgent)

    st.markdown("**Répartition par catégorie métier**")
    st.plotly_chart(chart_categories(df), use_container_width=True)

    st.markdown("**Urgence des délais**")
    st.plotly_chart(chart_urgence(df), use_container_width=True)

    st.markdown("**Top 10 acheteurs publics**")
    st.plotly_chart(chart_top_acheteurs(df), use_container_width=True)

    st.markdown("**Évolution des publications**")
    fig_evolution = chart_evolution(df)
    if fig_evolution is not None:
        st.plotly_chart(fig_evolution, use_container_width=True)
    else:
        st.caption("Dates de publication non disponibles.")


# ------------------------------------------------------------------------------
# PAGE
# ------------------------------------------------------------------------------

st.title("📊 Dashboard")
st.caption("Vue d'ensemble des offres actives — Avis A.O et Shopping Mall, côte à côte.")
st.divider()

col_ao, col_sm = st.columns(2)

with col_ao:
    render_volet(df_avis_ao, "Avis A.O (> 100 000 DT)", "📁")

with col_sm:
    render_volet(df_shopping_mall, "Shopping Mall (< 100 000 DT)", "🛒")
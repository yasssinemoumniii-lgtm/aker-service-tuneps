"""
================================================================================
 AGENT 2 — CLEANER / NORMALIZER (Aker Service)
================================================================================
Rôle dans le pipeline multi-agents :
    Agent1Scraper  -> app/data/raw_avis_ao.json + raw_shopping_mall.json
    Agent2Cleaner  -> LIT ces fichiers bruts, nettoie et normalise, écrit :
                         app/data/cleaned_avis_ao.json
                         app/data/cleaned_shopping_mall.json
                         app/data/cleaned_combined.csv   (les deux volets réunis,
                                                           schéma commun, prêt pour
                                                           Agent3 ML)
    Agent3ML       -> lira cleaned_combined.csv pour le clustering K-means

Étapes de nettoyage (validées ensemble) :
    1. Chargement des JSON bruts
    2. Nettoyage du texte (espaces, guillemets/apostrophes mal encodés, mojibake)
       -> le texte reste mixte français/arabe tel quel, PAS de filtrage par langue
    3. Parsing des dates + calcul des jours restants avant échéance
    4. Suppression des doublons (même numéro d'AO / de consultation)
    5. Harmonisation en schéma commun (avis_ao + shopping_mall -> mêmes colonnes)
    6. Export (JSON par volet + CSV combiné)

Utilisation autonome (test) :
    python agents/agent2_cleaner.py

Utilisation depuis app.py :
    from agents.agent2_cleaner import Agent2Cleaner

    agent = Agent2Cleaner()
    result = agent.run()   # -> {"avis_ao": df, "shopping_mall": df, "combined": df}

Prérequis :
    pip install pandas ftfy
================================================================================
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import ftfy
    _HAS_FTFY = True
except ImportError:
    _HAS_FTFY = False

log = logging.getLogger("agent2_cleaner")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [Agent2-Cleaner] %(message)s",
        datefmt="%H:%M:%S",
    )


# ------------------------------------------------------------------------------
# NETTOYAGE DE TEXTE
# ------------------------------------------------------------------------------

def clean_text(text) -> str:
    """
    Nettoie une chaîne de texte (objet, acheteur public...) :
    - corrige le mojibake / mauvais encodage (via ftfy si dispo)
    - normalise les guillemets/apostrophes typographiques -> apostrophe simple
    - réduit les espaces multiples
    - ne touche PAS à la langue : le texte reste mixte français/arabe tel quel
    """
    if text is None:
        return ""
    text = str(text)

    if _HAS_FTFY:
        text = ftfy.fix_text(text)

    # Normalisation des apostrophes/guillemets typographiques
    replacements = {
        "\u2019": "'", "\u2018": "'",   # ' '
        "\u201c": '"', "\u201d": '"',   # " "
        "\u00a0": " ",                   # espace insécable -> espace normal
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Espaces multiples -> un seul espace, on trim
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------------------------------
# PARSING DE DATES
# ------------------------------------------------------------------------------

def parse_date(text: str):
    """Parse 'dd/mm/YYYY' ou 'dd/mm/YYYY HH:MM' -> datetime, sinon None."""
    if not text:
        return None
    text = text.strip()
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


# ------------------------------------------------------------------------------
# AGENT 2 : CLEANER
# ------------------------------------------------------------------------------

class Agent2Cleaner:
    """Agent responsable du nettoyage et de la normalisation des données scrapées."""

    def __init__(self, data_dir: str | Path = None):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "app" / "data"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- Étape 1 : chargement --------------------

    def _load_raw(self, filename: str) -> list[dict]:
        path = self.data_dir / filename
        if not path.exists():
            log.warning(f"Fichier introuvable : {path} (Agent1 a-t-il tourné avant ?)")
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log.info(f"Chargé : {path} ({len(data)} lignes brutes)")
        return data

    # -------------------- Étape 2+3 : nettoyage texte + dates --------------------

    def _clean_dataframe(self, df: pd.DataFrame, text_cols: list[str], date_cols: list[str]) -> pd.DataFrame:
        if df.empty:
            return df

        for col in text_cols:
            df[col] = df[col].apply(clean_text)

        for col in date_cols:
            parsed_col = f"{col}_dt"
            df[parsed_col] = df[col].apply(parse_date)

        # Jours restants avant l'échéance (basé sur 'dernier_delai')
        if "dernier_delai_dt" in df.columns:
            now = datetime.now()
            df["jours_restants"] = df["dernier_delai_dt"].apply(
                lambda d: (d - now).days if d is not None else None
            )

        return df

    # -------------------- Étape 4 : doublons --------------------

    def _drop_duplicates(self, df: pd.DataFrame, key_col: str) -> pd.DataFrame:
        if df.empty:
            return df
        before = len(df)
        df = df.drop_duplicates(subset=[key_col], keep="first").reset_index(drop=True)
        removed = before - len(df)
        if removed:
            log.info(f"  {removed} doublon(s) supprimé(s) sur la colonne '{key_col}'.")
        return df

    # -------------------- Étape 6 : export --------------------

    def _save_json(self, df: pd.DataFrame, filename: str):
        path = self.data_dir / filename
        # On ne garde pas les colonnes *_dt (objets datetime) dans le JSON, juste les infos utiles
        export_df = df.drop(columns=[c for c in df.columns if c.endswith("_dt")], errors="ignore")
        export_df.to_json(path, orient="records", force_ascii=False, indent=2)
        log.info(f"Sauvegardé : {path} ({len(df)} lignes)")

    def _save_csv(self, df: pd.DataFrame, filename: str):
        path = self.data_dir / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        log.info(f"Sauvegardé : {path} ({len(df)} lignes)")

    # -------------------- Pipeline complet --------------------

    def clean_avis_ao(self) -> pd.DataFrame:
        raw = self._load_raw("raw_avis_ao.json")
        df = pd.DataFrame(raw)
        if df.empty:
            log.warning("Aucune donnée Avis A.O à nettoyer.")
            return df

        df = self._clean_dataframe(
            df,
            text_cols=["acheteur_public", "objet_ao"],
            date_cols=["date_publication", "dernier_delai"],
        )
        df = self._drop_duplicates(df, key_col="numero_ao")
        log.info(f"Avis A.O nettoyé : {len(df)} lignes.")
        return df

    def clean_shopping_mall(self) -> pd.DataFrame:
        raw = self._load_raw("raw_shopping_mall.json")
        df = pd.DataFrame(raw)
        if df.empty:
            log.warning("Aucune donnée Shopping Mall à nettoyer.")
            return df

        df = self._clean_dataframe(
            df,
            text_cols=["acheteur_public", "objet_consultation"],
            date_cols=["date_publication", "dernier_delai"],
        )
        df = self._drop_duplicates(df, key_col="numero_consultation")
        log.info(f"Shopping Mall nettoyé : {len(df)} lignes.")
        return df

    def run(self) -> dict:
        """
        Point d'entrée principal : nettoie les deux volets et les exporte
        SÉPARÉMENT (JSON + CSV chacun), sans les fusionner. Agent3 fera un
        clustering indépendant pour chaque volet, et le dashboard Streamlit
        affichera les deux dans deux sections distinctes.
        """
        avis_ao = self.clean_avis_ao()
        self._save_json(avis_ao, "cleaned_avis_ao.json")
        self._save_csv(avis_ao.drop(columns=[c for c in avis_ao.columns if c.endswith("_dt")], errors="ignore"), "cleaned_avis_ao.csv")

        shopping_mall = self.clean_shopping_mall()
        self._save_json(shopping_mall, "cleaned_shopping_mall.json")
        self._save_csv(shopping_mall.drop(columns=[c for c in shopping_mall.columns if c.endswith("_dt")], errors="ignore"), "cleaned_shopping_mall.csv")

        log.info("Agent2Cleaner terminé.")
        return {"avis_ao": avis_ao, "shopping_mall": shopping_mall}


# ------------------------------------------------------------------------------
# TEST AUTONOME
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    agent = Agent2Cleaner()
    result = agent.run()
    print(f"Avis A.O nettoyé      : {len(result['avis_ao'])} lignes")
    print(f"Shopping Mall nettoyé : {len(result['shopping_mall'])} lignes")
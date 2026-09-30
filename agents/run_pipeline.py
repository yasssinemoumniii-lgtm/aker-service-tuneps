"""
================================================================================
 RUN_PIPELINE — Orchestrateur du pipeline Aker Service
================================================================================
Enchaîne automatiquement les 3 premiers agents, dans l'ordre :
    Agent1 (scraping)  ->  Agent2 (nettoyage)  ->  Agent3 (clustering, notebook)

Conçu pour être lancé automatiquement chaque jour via le Planificateur de
tâches Windows (voir run_pipeline.bat + les instructions fournies).

Chaque exécution écrit un log daté dans le dossier logs/, pour pouvoir
vérifier après coup que la mise à jour du jour s'est bien passée.

Utilisation manuelle (test) :
    python agents/run_pipeline.py

Prérequis additionnels : pip install jupyter nbconvert
================================================================================
"""

import logging
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENTS_DIR.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
NOTEBOOK_PATH = AGENTS_DIR / "agent_kmeans.ipynb"

# Le logging est configuré EN PREMIER, avant tout import fragile (selenium,
# pandas...), pour être certain qu'un fichier log soit écrit même si le
# script plante tout de suite (ex: mauvais interpréteur Python utilisé par
# la tâche planifiée, librairie manquante, etc.)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Pipeline] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("pipeline")
log.info(f"Interpréteur Python utilisé : {sys.executable}")

# Permet d'importer agent1_scraper / agent2_cleaner depuis ce même dossier
sys.path.insert(0, str(AGENTS_DIR))

try:
    from agent1_scraper import Agent1Scraper
    from agent2_cleaner import Agent2Cleaner
except Exception:
    log.error("Échec de l'import des agents — vérifiez que les librairies sont "
               "installées pour CET interpréteur Python précis :\n" + traceback.format_exc())
    raise


# ------------------------------------------------------------------------------
# ÉTAPES DU PIPELINE
# ------------------------------------------------------------------------------

def run_agent1():
    log.info("=== Agent 1 : Scraping ===")
    # Jusqu'à 2 tentatives complètes : si Chrome plante en cours de route
    # (session fermée, navigateur crashé), on relance avec un navigateur tout
    # neuf plutôt que de faire échouer tout le pipeline automatique.
    last_error = None
    for attempt in range(1, 3):
        try:
            agent = Agent1Scraper(headless=True)
            result = agent.run()
            log.info(f"Agent1 terminé : {len(result['avis_ao'])} Avis A.O actifs, {len(result['shopping_mall'])} Shopping Mall actifs")
            return
        except Exception as e:
            last_error = e
            log.warning(f"Agent1 a échoué (tentative {attempt}/2) : {e}")
            if attempt < 2:
                log.info("Nouvelle tentative avec un navigateur neuf dans 10s...")
                import time as _time
                _time.sleep(10)
    raise last_error


def run_agent2():
    log.info("=== Agent 2 : Nettoyage ===")
    agent = Agent2Cleaner()
    result = agent.run()
    log.info(f"Agent2 terminé : {len(result['avis_ao'])} / {len(result['shopping_mall'])} lignes nettoyées")


def run_agent3():
    log.info("=== Agent 3 : Clustering (exécution du notebook) ===")
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        "--ExecutePreprocessor.timeout=600",
        str(NOTEBOOK_PATH),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"Agent3 (notebook) a échoué :\n{result.stderr[-2000:]}")
        raise RuntimeError("Échec de l'exécution du notebook Agent3")
    log.info("Agent3 terminé.")


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    start = datetime.now()
    log.info(f"########## Démarrage du pipeline Aker Service — {start:%d/%m/%Y %H:%M} ##########")
    try:
        run_agent1()
        run_agent2()
        run_agent3()
        duree = datetime.now() - start
        log.info(f"########## Pipeline terminé avec succès en {duree} ##########")
    except Exception as e:
        log.error(f"########## Pipeline interrompu : {e} ##########")
        raise


if __name__ == "__main__":
    main()
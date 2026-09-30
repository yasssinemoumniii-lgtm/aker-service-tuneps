"""
================================================================================
 AGENT 1 — SCRAPER (Aker Service)
================================================================================
Rôle dans le pipeline multi-agents :
    Agent1Scraper  -> récupère les données brutes de TUNEPS (Avis A.O +
                       Shopping Mall) et les sauvegarde en JSON dans app/data/
    Agent2Cleaner  -> lira ces fichiers pour nettoyer/normaliser
    Agent3ML       -> exploitera les données nettoyées

Utilisation autonome (test) :
    python agents/agent1_scraper.py

Utilisation depuis app.py :
    from agents.agent1_scraper import Agent1Scraper

    agent = Agent1Scraper()
    result = agent.run()   # -> {"avis_ao": [...], "shopping_mall": [...]}

Prérequis :
    pip install selenium webdriver-manager
================================================================================
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    ElementClickInterceptedException,
)
from webdriver_manager.chrome import ChromeDriverManager

log = logging.getLogger("agent1_scraper")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [Agent1-Scraper] %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_deadline(text: str):
    """
    Parse une date de délai du type '27/02/2026 10:00' ou '27/02/2026'.
    Retourne un datetime, ou None si le format n'est pas reconnu.
    """
    if not text:
        return None
    text = text.strip()
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _is_active(deadline_text: str) -> bool:
    """Une offre est 'active' si son dernier délai n'est pas encore passé."""
    dt = _parse_deadline(deadline_text)
    if dt is None:
        # Une date manquante ou illisible est TRÈS fréquente sur les vieilles
        # archives TUNEPS (champ vide une fois l'offre classée). La traiter
        # comme "active par prudence" empêchait le scraping de jamais
        # s'arrêter (des centaines de pages d'archives comptées à tort comme
        # actives), jusqu'à faire planter Chrome. On la traite donc comme
        # NON active : plus sûr, et ça permet à l'arrêt anticipé de fonctionner.
        return False
    return dt >= datetime.now()


# ------------------------------------------------------------------------------
# MODÈLES DE DONNÉES
# ------------------------------------------------------------------------------

@dataclass
class AvisAO:
    numero_ao: str
    acheteur_public: str
    date_publication: str
    objet_ao: str
    dernier_delai: str
    categorie: str = "avis_ao"          # > 100 000 DT


@dataclass
class Consultation:
    numero_consultation: str
    acheteur_public: str
    date_publication: str
    objet_consultation: str
    dernier_delai: str
    categorie: str = "shopping_mall"    # < 100 000 DT


# ------------------------------------------------------------------------------
# AGENT 1 : SCRAPER
# ------------------------------------------------------------------------------

class Agent1Scraper:
    """Agent responsable du scraping complet du portail TUNEPS."""

    BASE_URL = "https://www.tuneps.tn/portail"
    URLS = {
        "avis_ao": f"{BASE_URL}/offres",
        "shopping_mall": f"{BASE_URL}/consultations",
    }

    def __init__(
        self,
        headless: bool = True,
        page_load_timeout: int = 45,
        wait_between_pages: float = 1.2,
        max_pages: int = 300,  # filet de sécurité : borne le pire des cas même si l'arrêt anticipé échoue
        data_dir: str | Path = None,
    ):
        self.headless = headless
        self.page_load_timeout = page_load_timeout
        self.wait_between_pages = wait_between_pages
        self.max_pages = max_pages

        # app/data/ par défaut (à côté du dossier agents/)
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "app" / "data"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.driver = None

    # -------------------- Cycle de vie du navigateur --------------------

    def _create_driver(self) -> webdriver.Chrome:
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
            # Options qui stabilisent Chrome headless (surtout pour un site
            # Angular lourd comme TUNEPS, et pour tourner via le Planificateur
            # de tâches Windows, sans session graphique complète) :
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--lang=fr-FR")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(self.page_load_timeout)
        return driver

    def __enter__(self):
        self.driver = self._create_driver()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.quit()
            self.driver = None

    # -------------------- Helpers de scraping --------------------

    def _wait_for_table(self, timeout: int = 20):
        WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
        )
        WebDriverWait(self.driver, timeout).until(
            lambda d: len(d.find_elements(By.CSS_SELECTOR, "table tbody tr")) > 0
        )

    def _extract_rows(self, model_cls, field_names: list[str]) -> list:
        rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        results = []
        for row in rows:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) < len(field_names):
                    continue
                values = [cells[i].text.strip() for i in range(len(field_names))]
                results.append(model_cls(**dict(zip(field_names, values))))
            except StaleElementReferenceException:
                continue
        return results

    def _is_disabled(self, el) -> bool:
        try:
            classes = (el.get_attribute("class") or "").lower()
            aria_disabled = (el.get_attribute("aria-disabled") or "").lower()
            disabled_attr = el.get_attribute("disabled")
            parent_classes = ""
            try:
                parent_classes = (
                    el.find_element(By.XPATH, "..").get_attribute("class") or ""
                ).lower()
            except NoSuchElementException:
                pass
            return (
                "disabled" in classes
                or "disabled" in parent_classes
                or aria_disabled == "true"
                or disabled_attr is not None
            )
        except StaleElementReferenceException:
            # L'élément a disparu entre-temps (Angular a redessiné la page) :
            # on le considère prudemment comme indisponible plutôt que de planter.
            return True

    def _find_next_by_paginator_structure(self):
        """
        Repère la pagination via le libellé du type "5121 - 5130 de 58899"
        (motif typique Angular Material / mat-paginator), puis remonte dans
        le DOM jusqu'à trouver un conteneur avec au moins 2 boutons.
        Le dernier bouton actif de ce conteneur = 'suivant'.
        Fonctionne même si les boutons ‹ › sont des icônes sans texte/aria-label.
        """
        range_pattern = re.compile(r"\d[\d\s]*[-–]\s*\d[\d\s]*\s*(de|sur|of)\s*\d[\d\s]*", re.IGNORECASE)

        label_el = None
        try:
            candidates = self.driver.find_elements(By.XPATH, "//*[not(self::script or self::style)]")
        except Exception:
            return None

        for el in candidates:
            try:
                txt = el.text
            except StaleElementReferenceException:
                continue
            if txt and len(txt) < 40 and range_pattern.search(txt):
                label_el = el
                break

        if label_el is None:
            return None

        current = label_el
        for _ in range(6):
            try:
                current = current.find_element(By.XPATH, "..")
            except (NoSuchElementException, StaleElementReferenceException):
                break
            pool = current.find_elements(By.CSS_SELECTOR, "button, a[role='button'], [role='button']")
            if len(pool) >= 2:
                enabled = []
                for b in pool:
                    try:
                        if b.is_displayed() and not self._is_disabled(b):
                            enabled.append(b)
                    except StaleElementReferenceException:
                        continue
                if enabled:
                    return enabled[-1]  # dernier bouton du bloc = 'suivant' (‹ est avant ›)
        return None

    def set_max_page_size(self):
        """
        Best-effort : essaie d'augmenter le nombre de lignes par page (souvent
        un mat-select 'lignes par page' à côté de la pagination) pour réduire
        drastiquement le nombre de pages à parcourir. Si ça échoue, on continue
        simplement avec la taille de page par défaut (aucun impact bloquant).
        """
        try:
            # Cherche un élément affichant juste un petit nombre (10, 25...) proche de la pagination,
            # généralement un mat-select fermé.
            size_candidates = self.driver.find_elements(
                By.XPATH,
                "//*[self::mat-select or contains(@class,'select') or contains(@role,'listbox')]"
            )
            for el in size_candidates:
                txt = (el.text or "").strip()
                if txt.isdigit() and int(txt) <= 100:
                    el.click()
                    time.sleep(0.5)
                    options = self.driver.find_elements(
                        By.CSS_SELECTOR, "mat-option, [role='option'], li"
                    )
                    numeric_options = [
                        (int(o.text.strip()), o) for o in options if o.text.strip().isdigit()
                    ]
                    if numeric_options:
                        # On ne prend PAS systématiquement la plus grande taille disponible
                        # (souvent 800+) : un tableau aussi énorme fait grimper la mémoire
                        # de Chrome et peut le faire planter sur une session longue.
                        # On plafonne à MAX_PAGE_SIZE_CAP, tout en gardant le principe de
                        # réduire le nombre de pages à parcourir.
                        MAX_PAGE_SIZE_CAP = 100
                        options_sous_plafond = [o for o in numeric_options if o[0] <= MAX_PAGE_SIZE_CAP]
                        candidats_tries = sorted(
                            options_sous_plafond or numeric_options, key=lambda x: x[0], reverse=True
                        )
                        best_value, best_el = candidats_tries[0]
                        rows_before = len(self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr"))
                        best_el.click()

                        # On attend que le nombre de lignes affichées augmente VRAIMENT
                        # (jusqu'à 10s) — un simple "table présente" ne suffit pas, Angular
                        # peut encore afficher l'ancien tableau (10 lignes) un court instant.
                        try:
                            WebDriverWait(self.driver, 10).until(
                                lambda d: len(d.find_elements(By.CSS_SELECTOR, "table tbody tr")) > rows_before
                                or len(d.find_elements(By.CSS_SELECTOR, "table tbody tr")) >= best_value
                            )
                        except TimeoutException:
                            log.warning(
                                f"  Le tableau ne semble pas s'être mis à jour après le changement "
                                f"de taille de page (toujours {rows_before} lignes) — on continue quand même."
                            )

                        log.info(f"  Taille de page augmentée à {best_value} lignes/page.")
                        return True
        except Exception as e:
            log.debug(f"  set_max_page_size a échoué (sans impact) : {e}")
        return False

    def _find_next_button(self):
        # Niveau 0 : approche structurelle (marche même pour boutons icône sans texte)
        structural = self._find_next_by_paginator_structure()
        if structural is not None:
            return structural
        """
        Stratégie en 4 niveaux, du plus spécifique au plus générique :
          1) sélecteurs CSS connus (Angular Material / PrimeNG / Bootstrap)
          2) recherche par texte/aria-label contenant 'suivant' / 'next'
          3) recherche par symbole de flèche (›, », >)
          4) pagination numérotée : trouver la page active, cliquer le numéro suivant
        """
        # Scroll en bas pour s'assurer que la pagination (souvent sous le tableau) est chargée
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.3)

        # --- Niveau 1 : sélecteurs CSS connus ---
        candidate_selectors = [
            "a[aria-label='Next']",
            "a[aria-label='Suivant']",
            "li.pagination-next a",
            "li.next a",
            "button[aria-label='Next Page']",
            ".p-paginator-next",
            "a.page-link[rel='next']",
            "button.page-link",
            "a.page-link",
            "[class*='pagination'] a",
            "[class*='pagination'] button",
            "[class*='paginator'] a",
            "[class*='paginator'] button",
        ]
        for selector in candidate_selectors:
            for el in self.driver.find_elements(By.CSS_SELECTOR, selector):
                text = (el.text or "").strip().lower()
                aria = (el.get_attribute("aria-label") or "").strip().lower()
                if ("suivant" in text or "next" in text or "suivant" in aria
                        or "next" in aria or text in (">", "›", "»")):
                    if el.is_displayed() and not self._is_disabled(el):
                        return el

        # --- Niveau 2 : XPath texte/aria-label 'suivant'/'next' ---
        xpath_text = (
            "//a[contains(translate(., 'SUIVANTNEXT', 'suivantnext'), 'suivant') "
            "or contains(translate(., 'SUIVANTNEXT', 'suivantnext'), 'next') "
            "or contains(translate(@aria-label,'SUIVANTNEXT','suivantnext'), 'suivant') "
            "or contains(translate(@aria-label,'SUIVANTNEXT','suivantnext'), 'next')] | "
            "//button[contains(translate(., 'SUIVANTNEXT', 'suivantnext'), 'suivant') "
            "or contains(translate(., 'SUIVANTNEXT', 'suivantnext'), 'next') "
            "or contains(translate(@aria-label,'SUIVANTNEXT','suivantnext'), 'suivant') "
            "or contains(translate(@aria-label,'SUIVANTNEXT','suivantnext'), 'next')]"
        )
        for el in self.driver.find_elements(By.XPATH, xpath_text):
            if el.is_displayed() and not self._is_disabled(el):
                return el

        # --- Niveau 3 : symboles de flèche ---
        for symbol in [">", "›", "»", "▶"]:
            xpath_symbol = f"//a[normalize-space(text())='{symbol}'] | //button[normalize-space(text())='{symbol}'] | //span[normalize-space(text())='{symbol}']"
            for el in self.driver.find_elements(By.XPATH, xpath_symbol):
                clickable = el
                # si c'est un <span>, le vrai élément cliquable est souvent le parent <a>/<button>
                tag = el.tag_name.lower()
                if tag not in ("a", "button"):
                    try:
                        parent = el.find_element(By.XPATH, "./ancestor::*[self::a or self::button][1]")
                        clickable = parent
                    except NoSuchElementException:
                        continue
                if clickable.is_displayed() and not self._is_disabled(clickable):
                    return clickable

        # --- Niveau 4 : pagination numérotée (trouver la page active, cliquer sur la suivante) ---
        active_selectors = [
            "[class*='active']", "[class*='current']", "[aria-current='page']",
        ]
        for selector in active_selectors:
            for el in self.driver.find_elements(By.CSS_SELECTOR, selector):
                text = (el.text or "").strip()
                if text.isdigit():
                    try:
                        next_num = str(int(text) + 1)
                        xpath_next_num = (
                            f"//a[normalize-space(text())='{next_num}'] | "
                            f"//button[normalize-space(text())='{next_num}'] | "
                            f"//li[normalize-space(text())='{next_num}']"
                        )
                        candidates = self.driver.find_elements(By.XPATH, xpath_next_num)
                        for cand in candidates:
                            if cand.is_displayed() and not self._is_disabled(cand):
                                return cand
                    except ValueError:
                        continue

        return None

    def debug_pagination(self):
        """
        Diagnostic : scanne la page à la recherche de TOUT ce qui ressemble
        à de la pagination, et sauvegarde le résultat dans un fichier texte
        (app/data/debug_pagination.txt) à m'envoyer si find_next_button échoue.
        """
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.3)

        lines = ["=== DIAGNOSTIC PAGINATION ===", f"URL: {self.driver.current_url}", ""]

        # 1) Tout élément dont la classe contient 'pag'
        lines.append("--- Éléments avec classe contenant 'pag' ---")
        for el in self.driver.find_elements(By.CSS_SELECTOR, "[class*='pag']"):
            try:
                lines.append(
                    f"<{el.tag_name} class='{el.get_attribute('class')}'> "
                    f"text='{el.text.strip()[:60]}' outerHTML(200)='{el.get_attribute('outerHTML')[:200]}'"
                )
            except Exception:
                pass

        # 2) Tout lien/bouton contenant un chiffre seul (ex: numéros de page)
        lines.append("")
        lines.append("--- Liens/boutons avec du texte numérique court ---")
        for el in self.driver.find_elements(By.CSS_SELECTOR, "a, button"):
            txt = (el.text or "").strip()
            if txt.isdigit() or txt in (">", "›", "»", "<", "‹", "«"):
                try:
                    lines.append(
                        f"<{el.tag_name} class='{el.get_attribute('class')}'> "
                        f"text='{txt}' aria-label='{el.get_attribute('aria-label')}'"
                    )
                except Exception:
                    pass

        debug_path = self.data_dir / "debug_pagination.txt"
        debug_path.write_text("\n".join(lines), encoding="utf-8")
        log.info(f"Diagnostic pagination sauvegardé : {debug_path}")
        return debug_path

    def _click_next_page(self, next_btn) -> bool:
        try:
            old_first_row = ""
            rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
            if rows:
                old_first_row = rows[0].text

            self.driver.execute_script(
                "arguments[0].scrollIntoView({block:'center'});", next_btn
            )
            try:
                next_btn.click()
            except ElementClickInterceptedException:
                self.driver.execute_script("arguments[0].click();", next_btn)

            WebDriverWait(self.driver, 15).until(
                lambda d: (
                    len(d.find_elements(By.CSS_SELECTOR, "table tbody tr")) > 0
                    and d.find_elements(By.CSS_SELECTOR, "table tbody tr")[0].text
                    != old_first_row
                )
            )
            time.sleep(self.wait_between_pages)
            return True
        except TimeoutException:
            return False
        except StaleElementReferenceException:
            # Le bouton (ou la ligne de référence) a disparu entre-temps car
            # Angular a redessiné la page pile à ce moment — pas une vraie
            # erreur, juste un mauvais timing. On traite ça comme "le clic
            # n'a rien changé" ; l'appelant retentera sur la page suivante.
            log.warning("  Élément devenu obsolète pendant le clic (page redessinée), nouvelle tentative...")
            return False

    def _scrape_section(self, url: str, model_cls, field_names: list[str], active_only: bool = True) -> list:
        log.info(f"Ouverture : {url}")

        # Jusqu'à 3 tentatives si la page timeout au chargement (fréquent en
        # headless sur un site Angular lourd) — évite qu'un simple ralentissement
        # réseau fasse échouer tout le pipeline automatique.
        for attempt in range(1, 4):
            try:
                self.driver.get(url)
                self._wait_for_table()
                break
            except TimeoutException:
                log.warning(f"  Timeout au chargement (tentative {attempt}/3)...")
                if attempt == 3:
                    raise
                time.sleep(5)

        self.set_max_page_size()  # best-effort, réduit le nombre de pages à parcourir

        all_results = []
        page_num = 1
        consecutive_pages_without_active = 0

        while True:
            log.info(f"  Page {page_num}...")
            page_results = self._extract_rows(model_cls, field_names)
            log.info(f"    {len(page_results)} lignes.")

            if active_only:
                active_rows = [r for r in page_results if _is_active(r.dernier_delai)]
                log.info(f"    dont {len(active_rows)} encore actives.")
                all_results.extend(active_rows)

                if len(active_rows) == 0 and len(page_results) > 0:
                    consecutive_pages_without_active += 1
                else:
                    consecutive_pages_without_active = 0

                # Le tableau est trié par date de publication décroissante :
                # après 2 pages consécutives sans aucune offre active, tout ce
                # qui suit est forcément encore plus ancien -> on arrête.
                if consecutive_pages_without_active >= 2:
                    log.info(
                        "  Plus aucune offre active depuis 2 pages : "
                        "arrêt anticipé (le reste est expiré)."
                    )
                    break
            else:
                all_results.extend(page_results)

            if page_num >= self.max_pages:
                log.warning("max_pages atteint, arrêt.")
                break

            clicked = False
            for click_attempt in range(1, 3):  # jusqu'à 2 tentatives complètes
                next_btn = self._find_next_button()
                if next_btn is None:
                    break
                if self._click_next_page(next_btn):
                    clicked = True
                    break
                log.info(f"  Clic sans effet, nouvelle recherche du bouton (tentative {click_attempt}/2)...")
                time.sleep(2)

            if not clicked:
                if page_num == 1 and next_btn is None:
                    log.warning(
                        "  Aucun bouton de pagination détecté sur la page 1 : "
                        "lancement du diagnostic automatique..."
                    )
                    self.debug_pagination()
                log.info("  Fin de la pagination.")
                break
            page_num += 1

        return all_results

    # -------------------- Méthodes publiques (utilisées par app.py) --------------------

    def scrape_avis_ao(self, active_only: bool = True) -> list[AvisAO]:
        """Scrape le volet 'Avis A.O' (> 100 000 DT). Par défaut, ne garde que les offres actives."""
        fields = ["numero_ao", "acheteur_public", "date_publication", "objet_ao", "dernier_delai"]
        return self._scrape_section(self.URLS["avis_ao"], AvisAO, fields, active_only=active_only)

    def scrape_shopping_mall(self, active_only: bool = True) -> list[Consultation]:
        """Scrape le volet 'Shopping Mall' (< 100 000 DT). Par défaut, ne garde que les offres actives."""
        fields = [
            "numero_consultation", "acheteur_public", "date_publication",
            "objet_consultation", "dernier_delai",
        ]
        return self._scrape_section(self.URLS["shopping_mall"], Consultation, fields, active_only=active_only)

    def _save_json(self, data: list, filename: str):
        path = self.data_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(d) for d in data], f, ensure_ascii=False, indent=2)
        log.info(f"Sauvegardé : {path} ({len(data)} lignes)")

    def run(self) -> dict:
        """
        Point d'entrée principal de l'agent : scrape les deux volets,
        sauvegarde en JSON dans app/data/, et retourne les données
        pour que app.py les transmette directement à Agent2Cleaner.
        """
        with self:
            avis_ao = self.scrape_avis_ao()
            self._save_json(avis_ao, "raw_avis_ao.json")

            shopping_mall = self.scrape_shopping_mall()
            self._save_json(shopping_mall, "raw_shopping_mall.json")

        log.info("Agent1Scraper terminé.")
        return {
            "avis_ao": [asdict(d) for d in avis_ao],
            "shopping_mall": [asdict(d) for d in shopping_mall],
        }


# ------------------------------------------------------------------------------
# TEST AUTONOME
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # headless=False pour voir le navigateur pendant les tests / le débogage
    agent = Agent1Scraper(headless=False)
    result = agent.run()
    print(f"Avis A.O       : {len(result['avis_ao'])} lignes")
    print(f"Shopping Mall  : {len(result['shopping_mall'])} lignes")

# ==============================================================================
# SI ÇA NE MARCHE PAS (0 ligne trouvée, ou boucle infinie sur la pagination)
# ==============================================================================
# Voir la note détaillée envoyée avec le premier script : les sélecteurs de
# pagination (_find_next_button) sont les plus probables mais pas garantis
# sans avoir inspecté le DOM réel. Ouvrez F12 sur tuneps.tn -> Elements,
# repérez le bouton "suivant" de la pagination, et ajoutez son sélecteur
# exact en tête de la liste candidate_selectors ci-dessus.
#
# Encore mieux : F12 -> Network -> XHR -> changez de page sur le vrai site.
# Si une requête API apparaît (ex: /api/offres?page=2&size=20), envoyez-la
# moi : on remplace Selenium par de simples appels `requests`, bien plus
# rapides et stables pour un agent qui doit tourner régulièrement.
# ==============================================================================
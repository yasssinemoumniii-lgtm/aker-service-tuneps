@echo off
REM ==============================================================================
REM  Lance le pipeline complet Aker Service (Agent1 -> Agent2 -> Agent3)
REM  A utiliser avec le Planificateur de taches Windows pour une mise a jour
REM  automatique quotidienne.
REM ==============================================================================

REM Se placer a la racine du projet (le dossier qui contient ce fichier .bat)
cd /d "%~dp0"

REM S'assurer que le dossier logs/ existe, meme si run_pipeline.py plante
REM avant d'avoir pu le creer lui-meme (ex: mauvais Python, erreur immediate).
if not exist logs mkdir logs

REM IMPORTANT : remplacez la ligne ci-dessous par le chemin COMPLET vers le
REM python.exe qui a bien selenium/pandas/etc. installes (celui utilise dans
REM VS Code). Pour le trouver : ouvrez un terminal VS Code avec le bon
REM interpreteur selectionne, et tapez "where python" - copiez le premier
REM chemin affiche.
set PYTHON_EXE=C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe
REM Tout ce qui s'affiche (y compris les erreurs Windows type "introuvable")
REM est capture dans ce fichier, meme si run_pipeline.py ne demarre jamais.
"%PYTHON_EXE%" agents\run_pipeline.py >> logs\bat_fallback.log 2>&1
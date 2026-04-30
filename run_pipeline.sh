#!/usr/bin/env bash
# Pipeline MLOps complet — Spotify Ads Prediction
# Usage : bash run_pipeline.sh
set -e  # arrête le script à la première erreur

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ─── Couleurs ────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

step() { echo -e "\n${CYAN}══════════════════════════════════════════${NC}"; \
         echo -e "${GREEN}▶ $1${NC}"; \
         echo -e "${CYAN}══════════════════════════════════════════${NC}\n"; }

warn() { echo -e "${YELLOW}⚠  $1${NC}"; }

# ─── 0. Venv ─────────────────────────────────────────────────────────────────
step "0. Activation du venv"
if [ ! -d ".venv" ]; then
    echo "Création du venv avec uv..."
    uv venv
fi
source .venv/Scripts/activate
echo "Python : $(python --version) — $(which python)"

# ─── 1. Dépendances ──────────────────────────────────────────────────────────
step "1. Installation des dépendances"
uv pip install -r requirements.txt
pip install -e . --quiet
echo "Package spotify_mlops installé."

# ─── 2. Données Kaggle ───────────────────────────────────────────────────────
step "2. Téléchargement du dataset Kaggle"
if [ -f "data/raw/spotify_churn_dataset.csv" ]; then
    warn "Dataset déjà présent (data/raw/spotify_churn_dataset.csv) — téléchargement ignoré."
else
    if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
        warn "Variables KAGGLE_USERNAME et KAGGLE_KEY non définies."
        warn "Assure-toi que ~/.kaggle/kaggle.json existe, ou définis les variables :"
        warn "  export KAGGLE_USERNAME=ton_username"
        warn "  export KAGGLE_KEY=ta_clé_api"
    fi
    mkdir -p data/raw
    kaggle datasets download \
        -d nabihazahid/spotify-dataset-for-churn-analysis \
        -p data/raw/ --unzip
    echo "Dataset téléchargé dans data/raw/"
fi

# ─── 3. Prétraitement ────────────────────────────────────────────────────────
step "3. Prétraitement (split 70/15/15 + encoding)"
python src/preprocess.py

# ─── 4. Entraînement ─────────────────────────────────────────────────────────
step "4. Entraînement (GridSearchCV sur 3 modèles + MLflow)"
python src/train.py

# ─── 5. Évaluation ───────────────────────────────────────────────────────────
step "5. Évaluation finale (test set + plots)"
python src/evaluate.py
echo "Plots générés → reports/evaluation_plots.png"

# ─── 6. Réentraînement batch val ─────────────────────────────────────────────
step "6. Réentraînement — batch val (simulation nouvelles données)"
python src/retrain.py --batch val

# ─── 7. Réentraînement batch test ────────────────────────────────────────────
step "7. Réentraînement — batch test (simulation nouvelles données)"
python src/retrain.py --batch test

# ─── 8. Résumé ───────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}✔  Pipeline terminé avec succès !${NC}"
echo ""
echo "  Prochaines étapes :"
echo "    mlflow ui                          → http://localhost:5000 (expériences)"
echo "    uvicorn src.serve:app --reload     → http://localhost:8000 (API)"
echo "    curl -X POST localhost:8000/reload → recharge le modèle dans l'API"
echo ""
echo "  Ou tout en un via Docker :"
echo "    docker-compose up --build"
echo ""

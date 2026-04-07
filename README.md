# Spotify Listening Time — MLOps Pipeline

Prédiction du temps d'écoute Spotify (`listening_time`) via régression linéaire.  
Projet MLOps Ynov 2025-2026.

## Stack

- **ML** : scikit-learn (LinearRegression, Ridge, Lasso + GridSearchCV)
- **Tracking** : MLflow
- **API** : FastAPI + Uvicorn
- **Conteneurisation** : Docker + Docker Compose
- **CI/CD** : GitHub Actions
- **Data** : Parquet (local)

## Structure

```
.
├── src/
│   ├── __init__.py       # package spotify_mlops
│   ├── preprocess.py     # split 70/15/15 + StandardScaler + OneHotEncoder → Parquet + npy
│   ├── train.py          # GridSearchCV sur 3 modèles + MLflow logging
│   ├── evaluate.py       # plots Actual vs Predicted + résidus
│   ├── retrain.py        # réentraînement avec nouvelles données (val ou test)
│   └── serve.py          # API REST FastAPI
├── notebooks/
│   ├── 01_eda.ipynb      # exploration du dataset
│   └── 02_full_pipeline.ipynb  # pipeline complet visuel
├── data/
│   ├── raw/              # données brutes (non versionné)
│   └── processed/        # Parquet + npy (non versionné)
├── models/               # modèle sérialisé (non versionné)
├── .github/workflows/    # CI/CD GitHub Actions
├── Dockerfile
├── docker-compose.yml
├── setup.py              # package installable
├── requirements.txt
└── Makefile
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .          # installe le package spotify_mlops
```

## Pipeline complet

```bash
# 1. Preprocessing (split 70/15/15 + export Parquet)
python src/preprocess.py

# 2. Entraînement (GridSearch + MLflow)
python src/train.py

# 3. Évaluation (plots)
python src/evaluate.py

# 4. Simulation nouvelles données — batch 1 (val)
python src/retrain.py --batch val

# 5. Simulation nouvelles données — batch 2 (test)
python src/retrain.py --batch test
```

Via Makefile :
```bash
make preprocess
make train
make evaluate
make retrain-val    # retrain avec val
make retrain-test   # retrain avec test
```

## API REST

```bash
# Local
uvicorn src.serve:app --reload
# → http://127.0.0.1:8000/docs

# Docker
docker-compose up --build
```

Exemple de requête :
```json
POST /predict
{
  "gender": "Female",
  "age": 28,
  "country": "FR",
  "subscription_type": "Premium",
  "songs_played_per_day": 20,
  "skip_rate": 0.15,
  "device_type": "Mobile",
  "ads_listened_per_week": 0,
  "offline_listening": 1
}
```

## MLflow UI

```bash
mlflow ui
# → http://127.0.0.1:5000
```

## CI/CD

Le workflow `.github/workflows/ml_pipeline.yml` se déclenche à chaque push sur `main` :
1. Installation des dépendances
2. Téléchargement du dataset (via secrets `KAGGLE_USERNAME` + `KAGGLE_KEY`)
3. Preprocessing
4. Entraînement
5. Upload du modèle comme artefact GitHub

**Secrets à configurer dans le repo GitHub :**
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

## Réentraînement automatique

Le dataset est divisé en 3 :
- **70%** — entraînement initial
- **15% (val)** — simule une 1ère vague de nouvelles données
- **15% (test)** — simule une 2ème vague de nouvelles données

Chaque `retrain.py` fusionne les nouvelles données avec le train existant, relance un GridSearch et met à jour le modèle servi par l'API.

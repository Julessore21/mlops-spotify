# Spotify Ads Prediction — Pipeline MLOps

Projet MLOps Ynov 2025-2026.  
Objectif : prédire le nombre de publicités écoutées par semaine (`ads_listened_per_week`) à partir du profil d'écoute d'un utilisateur Spotify.

---

## Outils utilisés

| Catégorie | Outil |
|---|---|
| Données | Kaggle (`nabihazahid/spotify-dataset-for-churn-analysis`) |
| Prétraitement | scikit-learn (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`) |
| Entraînement | scikit-learn (`LinearRegression`, `Ridge`, `Lasso`, `GridSearchCV`) |
| Tracking | MLflow |
| API REST | FastAPI + Uvicorn |
| Conteneurisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Stockage cloud | Azure Blob Storage |
| Package Python | `spotify_mlops` (installable via `pip install -e .`) |

---

## Structure du projet

```
mlops-spotify/
├── src/
│   ├── __init__.py       # package spotify_mlops
│   ├── preprocess.py     # split 70/15/15 + StandardScaler + OneHotEncoder → Parquet + npy
│   ├── train.py          # GridSearchCV sur 3 modèles + MLflow logging
│   ├── evaluate.py       # évaluation finale sur test set + plots + MLflow
│   ├── retrain.py        # réentraînement avec nouvelles données (val ou test)
│   └── serve.py          # API REST FastAPI (/predict, /reload, /health)
├── notebooks/
│   ├── 01_eda.ipynb            # exploration du dataset
│   ├── 02_full_pipeline.ipynb  # pipeline complet visuel
│   └── 03_mlops_story.ipynb    # démonstration narrative MLOps
├── data/
│   ├── raw/              # données brutes (non versionné)
│   └── processed/        # Parquet + npy (non versionné)
├── models/               # modèle sérialisé (non versionné)
├── reports/              # graphiques d'évaluation
├── .github/workflows/
│   ├── ml_pipeline.yml   # CI/CD déclenché à chaque push
│   └── retrain.yml       # réentraînement planifié (lundi 2h UTC)
├── Dockerfile
├── docker-compose.yml
├── setup.py              # package installable
├── requirements.txt
└── Makefile
```

---

## Étapes du projet

### 1. Collecte de données

Dataset public Kaggle : [Spotify Dataset for Churn Analysis](https://www.kaggle.com/datasets/nabihazahid/spotify-dataset-for-churn-analysis).

Contient ~10 000 utilisateurs avec les features : `gender`, `age`, `country`, `subscription_type`, `songs_played_per_day`, `skip_rate`, `device_type`, `listening_time`, `offline_listening`.  
Variable cible : `ads_listened_per_week`.

Téléchargement :
```bash
kaggle datasets download -d nabihazahid/spotify-dataset-for-churn-analysis -p data/raw/ --unzip
```

### 2. Prétraitement des données

**Script :** `src/preprocess.py`

Le dataset est divisé en **3 parties** pour simuler l'arrivée de nouvelles données au fil du temps :

| Split | Taille | Usage |
|---|---|---|
| `train.parquet` | 70 % | Entraînement initial |
| `val.parquet` | 15 % | Simulation 1ère vague de nouvelles données |
| `test.parquet` | 15 % | Simulation 2ème vague de nouvelles données |

Le prétraitement est réalisé via un pipeline scikit-learn (`ColumnTransformer`) :
- `StandardScaler` sur les variables numériques
- `OneHotEncoder` sur les variables catégorielles

Le preprocesseur est fitté **uniquement sur le train set** puis appliqué aux autres splits (pas de data leakage).

Les splits sont sauvegardés en **Parquet** (pour traçabilité et réentraînement) et en **NumPy arrays** (pour l'entraînement rapide). Ils sont également uploadés dans un **bucket Azure Blob Storage** si les variables d'environnement sont configurées.

```bash
python src/preprocess.py
```

### 3. Entraînement du modèle

**Script :** `src/train.py`

Trois modèles sont entraînés avec `GridSearchCV` (5-fold CV, scoring R²) :

| Modèle | Hyperparamètres explorés |
|---|---|
| `LinearRegression` | `fit_intercept`: [True, False] |
| `Ridge` | `alpha`: [0.1, 1.0, 10.0, 100.0], `fit_intercept`: [True, False] |
| `Lasso` | `alpha`: [0.01, 0.1, 1.0, 10.0], `fit_intercept`: [True, False] |

Pour chaque run, MLflow enregistre :
- Les hyperparamètres optimaux (`best_params`)
- Le score CV R² (`cv_best_r2`)
- Les métriques sur train et validation (RMSE, MAE, R²)
- Le modèle sérialisé comme artefact

Le meilleur modèle est sauvegardé dans `models/best_model.pkl`.

```bash
python src/train.py
mlflow ui  # → http://127.0.0.1:5000
```

**Reproductibilité :** le split est fixé avec `random_state=42`. Pour reproduire exactement une expérience, relancer `preprocess.py` puis `train.py` avec le même dataset.

### 4. Évaluation du modèle

**Script :** `src/evaluate.py`

L'évaluation finale est réalisée sur le **test set** (jamais vu pendant l'entraînement ni la sélection de modèle) :

| Métrique | Valeur |
|---|---|
| R² | 0.7726 |
| RMSE | 6.6636 |
| MAE | 3.0234 |

Meilleur modèle sélectionné : **Lasso** (`alpha=0.1`, `fit_intercept=True`, CV R²=0.7734).

> Les métriques complètes par run (train/val/test, tous modèles) sont disponibles dans MLflow UI (`mlflow ui`).

Les métriques sont loggées dans MLflow sous le run `final_evaluation`. Deux graphiques sont générés dans `reports/evaluation_plots.png` :
- Réel vs Prédit
- Graphique des résidus

```bash
python src/evaluate.py
```

### 5. Déploiement du modèle

**Script :** `src/serve.py` — API REST avec **FastAPI**.

Endpoints disponibles :

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/` | Message de bienvenue |
| `POST` | `/predict` | Prédiction à partir des features utilisateur |
| `POST` | `/reload` | Recharge le modèle depuis le disque (après réentraînement) |
| `GET` | `/health` | Health check |

Exemple de requête :
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "age": 28,
    "country": "France",
    "subscription_type": "Free",
    "songs_played_per_day": 20,
    "skip_rate": 0.15,
    "device_type": "Mobile",
    "listening_time": 120.5,
    "offline_listening": 0
  }'
# → {"predicted_ads_listened_per_week": 7.43}
```

**Démarrage local :**
```bash
uvicorn src.serve:app --reload
# → http://127.0.0.1:8000/docs
```

**Via Docker :**
```bash
docker-compose up --build
# API → http://localhost:8000
# MLflow → http://localhost:5000
```

L'image Docker embarque le modèle entraîné et expose le port 8000.

### 6. Réentraînement du modèle

**Script :** `src/retrain.py`

Quand de nouvelles données arrivent, le pipeline fusionne le train original avec le nouveau batch, relance un GridSearch et met à jour `best_model.pkl` :

```bash
# Simulation 1ère vague (val.parquet)
python src/retrain.py --batch val

# Simulation 2ème vague (test.parquet)
python src/retrain.py --batch test
```

Après réentraînement, l'API est mise à jour **sans redémarrage** via l'endpoint `/reload` :
```bash
curl -X POST http://localhost:8000/reload
# → {"status": "model reloaded"}
```

Le réentraînement est **planifié automatiquement** via GitHub Actions tous les lundis à 2h UTC (`.github/workflows/retrain.yml`), et peut aussi être déclenché manuellement avec le choix du batch.

### 7. CI/CD (Bonus)

**Workflow :** `.github/workflows/ml_pipeline.yml`

Déclenché à chaque push sur `main` ou `master` :

1. Checkout du code
2. Installation des dépendances (via `uv`)
3. Téléchargement du dataset Kaggle (secrets `KAGGLE_USERNAME` + `KAGGLE_KEY`)
4. Prétraitement (`preprocess.py`)
5. Entraînement (`train.py`)
6. Évaluation (`evaluate.py`) + upload des plots comme artefact GitHub
7. Upload du modèle comme artefact GitHub

**Secrets à configurer dans le repo GitHub :**
- `KAGGLE_USERNAME` / `KAGGLE_KEY` — accès Kaggle
- `AZURE_STORAGE_CONNECTION_STRING` / `AZURE_CONTAINER_NAME` — stockage cloud (optionnel)

---

## Installation et exécution complète

```bash
# Dépendances
pip install -r requirements.txt
pip install -e .          # installe le package spotify_mlops

# Pipeline complet
make preprocess           # split 70/15/15 + prétraitement
make train                # GridSearch + MLflow
make evaluate             # métriques finales + plots
make retrain-val          # réentraînement avec val (batch 1)
make retrain-test         # réentraînement avec test (batch 2)

# API
make serve                # uvicorn src.serve:app --reload
make mlflow-ui            # mlflow ui
```

---

## Package Python

Tout le code est packagé dans la librairie `spotify_mlops` :

```python
from spotify_mlops import preprocess, train, evaluate, retrain, serve
```

Installation :
```bash
pip install -e .
```

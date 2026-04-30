# Présentation Orale — Projet MLOps Spotify
## Prédiction des publicités écoutées par semaine

---

## INTRODUCTION (2 min)

### Contexte et objectif du projet

Ce projet s'inscrit dans le cadre du cours MLOps à Ynov. L'objectif est de construire un pipeline de machine learning complet, industrialisable et automatisé, en s'appuyant sur les bonnes pratiques de l'ingénierie logicielle appliquées au Machine Learning.

**Problème métier :** À partir du profil d'un utilisateur Spotify, prédire le nombre moyen de publicités qu'il écoute par semaine (`ads_listened_per_week`). C'est un problème de **régression supervisée**.

**Fichier à montrer :** [README.md](README.md) — présenter la section "Overview" et le schéma d'architecture général.

---

## PARTIE 1 — Les Données (3 min)

### 1.1 Source et description du dataset

Le dataset provient de Kaggle : `nabihazahid/spotify-dataset-for-churn-analysis`. Il contient **8 000 enregistrements** représentant des profils utilisateurs Spotify, initialement conçu pour la prédiction de churn (désabonnement), que l'on réutilise ici pour prédire les publicités.

**Fichier à montrer :** [data/raw/spotify_churn_dataset.csv](data/raw/spotify_churn_dataset.csv) — montrer les premières lignes pour illustrer les colonnes.

### 1.2 Les features utilisées

On dispose de **9 variables prédictives** réparties en deux types :

**Variables numériques (5) :**
- `age` — âge de l'utilisateur
- `songs_played_per_day` — nombre de chansons écoutées par jour
- `skip_rate` — taux de skip (proportion de chansons passées)
- `listening_time` — temps d'écoute quotidien
- `offline_listening` — indicateur d'écoute hors-ligne

**Variables catégorielles (4) :**
- `gender` — genre
- `country` — pays
- `subscription_type` — type d'abonnement (Free, Premium…)
- `device_type` — type d'appareil

**Variables supprimées intentionnellement :**
- `user_id` → identifiant sans valeur prédictive
- `is_churned` → variable cible d'une autre tâche, on l'exclut pour éviter tout biais

**Fichier à montrer :** [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) — montrer les cellules d'exploration : distributions, corrélations, valeurs manquantes.

---

## PARTIE 2 — Le Pipeline ML (10 min)

### 2.1 Prétraitement des données

**Fichier à montrer :** [src/preprocess.py](src/preprocess.py)

Le script `preprocess.py` effectue les opérations suivantes :

**a) Split des données** : on divise le dataset en trois ensembles distincts avec un ratio 70/15/15 et une graine aléatoire fixée (`random_state=42`) pour garantir la reproductibilité.

```
Train  : 70% → 5 600 exemples
Val    : 15% → 1 200 exemples
Test   : 15% → 1 200 exemples
```

**b) Pipeline de transformation** avec `ColumnTransformer` de scikit-learn :
- `StandardScaler` sur les variables numériques → centrage-réduction (moyenne=0, écart-type=1)
- `OneHotEncoder(handle_unknown="ignore")` sur les variables catégorielles → conversion en vecteurs binaires

**Point crucial à souligner :** Le preprocesseur est **fitté uniquement sur le train set**, puis appliqué sur val et test. Cela évite le **data leakage** (fuite d'information du futur vers le passé).

**c) Sorties du prétraitement :**
- Fichiers Parquet (`train.parquet`, `val.parquet`, `test.parquet`) → pour la traçabilité et le ré-entraînement
- Fichiers NumPy (`.npy`) → pour un chargement rapide pendant l'entraînement
- `preprocessor.pkl` → le transformateur sérialisé pour l'inférence en production

**Dossier à montrer :** [data/processed/](data/processed/) — montrer les fichiers générés.

---

### 2.2 Entraînement du modèle

**Fichier à montrer :** [src/train.py](src/train.py)

On utilise une **recherche exhaustive d'hyperparamètres** (`GridSearchCV`) sur trois familles de modèles linéaires, avec une **validation croisée à 5 folds** et la métrique R² comme critère de sélection.

**Modèles comparés :**

| Modèle | Hyperparamètres testés |
|---|---|
| LinearRegression | fit_intercept = [True, False] |
| Ridge (L2) | alpha = [0.1, 1.0, 10.0, 100.0] × fit_intercept |
| Lasso (L1) | alpha = [0.01, 0.1, 1.0, 10.0] × fit_intercept |

Pour chaque combinaison, un run MLflow est enregistré avec :
- Les hyperparamètres (`best_params_`)
- Le score CV (`cv_best_r2`)
- Les métriques sur train et val (RMSE, MAE, R²)
- Le modèle sérialisé avec sa signature et un exemple d'input

Le meilleur modèle est sauvegardé dans `models/best_model.pkl`.

**Fichier à montrer :** [models/best_model.pkl](models/best_model.pkl) — expliquer que c'est un objet Lasso sérialisé avec joblib.

---

### 2.3 Évaluation finale

**Fichier à montrer :** [src/evaluate.py](src/evaluate.py) puis [reports/evaluation_plots.png](reports/evaluation_plots.png)

L'évaluation se fait sur le **test set** (données jamais vues pendant l'entraînement ni la sélection de modèle).

**Résultats du meilleur modèle (Lasso, alpha=0.1) :**

| Métrique | Valeur |
|---|---|
| R² (test) | 0.7726 |
| RMSE (test) | 6.66 ads/semaine |
| MAE (test) | 3.02 ads/semaine |

**Interprétation :**
- Le modèle explique **77% de la variance** du nombre de publicités écoutées.
- En moyenne, la prédiction est juste à ±3 publicités par semaine.
- Le Lasso (régularisation L1) a été sélectionné car il introduit de la **sparsité** : il met à zéro les coefficients des features peu informatives.

**Graphiques à montrer :** [reports/evaluation_plots.png](reports/evaluation_plots.png)
- Graphique 1 : Valeurs réelles vs prédictions (idéalement alignées sur la diagonale)
- Graphique 2 : Résidus (erreurs centrées autour de zéro = bon signe)

---

### 2.4 Ré-entraînement sur nouvelles données

**Fichier à montrer :** [src/retrain.py](src/retrain.py)

Le script simule l'arrivée de **nouvelles données en production**. Le principe :

1. On charge l'ensemble d'entraînement original
2. On charge un nouveau batch de données (`val.parquet` ou `test.parquet`)
3. On transforme ce nouveau batch avec le preprocesseur déjà fitté
4. On **combine** les deux ensembles : `X_combined = [X_train + X_new_batch]`
5. On relance un GridSearchCV sur cet ensemble élargi
6. On sauvegarde le nouveau meilleur modèle dans `best_model.pkl`
7. Chaque run de ré-entraînement est loggué dans MLflow avec des tags (`retrain_batch`, `model_type`)

**Commande à montrer :**
```bash
python src/retrain.py --batch val
python src/retrain.py --batch test
```

---

## PARTIE 3 — Le Suivi d'Expériences avec MLflow (4 min)

### 3.1 Pourquoi MLflow ?

Sans outil de suivi, il est impossible de savoir quel modèle a été entraîné avec quels paramètres, ni de comparer plusieurs expériences. MLflow résout ce problème en centralisant toutes les informations.

**Interface à montrer en live :** http://localhost:5000

### 3.2 Ce qui est tracké

**Fichier à montrer :** [src/train.py](src/train.py) — montrer les appels `mlflow.log_param`, `mlflow.log_metric`, `mlflow.sklearn.log_model`

Pour chaque run MLflow, on enregistre :
- **Experiment** : `"spotify-ads-prediction"`
- **Run name** : ex. `"LinearRegression_gridsearch"`, `"Lasso_retrain_val"`
- **Params** : hyperparamètres du meilleur estimateur
- **Metrics** : `cv_best_r2`, `train_rmse`, `val_r2`, `test_mae`...
- **Artifacts** : modèle sérialisé avec signature + exemple d'input
- **Tags** : `model_type`, `retrain_batch`, hash de commit

**Dossier à montrer :** [mlruns/](mlruns/) — expliquer la structure hiérarchique : experiment → runs → params/metrics/artifacts.

### 3.3 Ce que l'on peut faire avec l'UI

- Comparer tous les runs côte à côte
- Filtrer par tag ou métrique
- Visualiser l'évolution des métriques au fil des ré-entraînements
- Télécharger un modèle versionné depuis n'importe quel run passé

---

## PARTIE 4 — L'API de Prédiction avec FastAPI (5 min)

### 4.1 Architecture de l'API

**Fichier à montrer :** [src/serve.py](src/serve.py)

L'API est construite avec **FastAPI** et exposée via **Uvicorn** (serveur ASGI asynchrone). Au démarrage, elle charge automatiquement `best_model.pkl` et `preprocessor.pkl` grâce au gestionnaire de cycle de vie (`lifespan`).

### 4.2 Les 4 endpoints

| Méthode | Route | Description |
|---|---|---|
| GET | `/` | Message de bienvenue |
| POST | `/predict` | Prédiction à partir des features utilisateur |
| POST | `/reload` | Rechargement à chaud du modèle après ré-entraînement |
| GET | `/health` | Vérification que l'API est opérationnelle |

### 4.3 Exemple de requête de prédiction

**À montrer en live** (avec l'API lancée sur http://localhost:8000) :

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 25,
    "gender": "Male",
    "country": "France",
    "subscription_type": "Free",
    "device_type": "Mobile",
    "songs_played_per_day": 15,
    "skip_rate": 0.3,
    "listening_time": 2.5,
    "offline_listening": 0
  }'
```

**Réponse attendue :**
```json
{
  "predicted_ads_listened_per_week": 12.47
}
```

### 4.4 Le rechargement à chaud

**Point fort à souligner :** Le endpoint `/reload` permet de **mettre à jour le modèle en production sans redémarrer l'API**. Après un ré-entraînement, il suffit d'appeler ce endpoint pour que l'API charge immédiatement le nouveau modèle.

```bash
# Après retrain.py, recharger le modèle sans couper le service :
curl -X POST http://localhost:8000/reload
```

### 4.5 Le middleware de logging

Chaque requête reçue est automatiquement logguée avec : méthode HTTP, route, code de statut, et temps de traitement. C'est une pratique standard de monitoring en production.

**Documentation auto à montrer :** http://localhost:8000/docs (Swagger UI généré automatiquement par FastAPI)

---

## PARTIE 5 — La Conteneurisation avec Docker (3 min)

### 5.1 Pourquoi Docker ?

Docker garantit que l'application s'exécute de manière **identique** sur tous les environnements (dev, CI, production), en encapsulant toutes les dépendances dans une image isolée.

**Fichier à montrer :** [Dockerfile](Dockerfile)

Le Dockerfile utilise l'image `python:3.10-slim` et `uv` (gestionnaire de paquets ultra-rapide) pour installer les dépendances. Il copie les artefacts (modèle + preprocesseur), expose le port 8000, et lance Uvicorn.

### 5.2 Docker Compose — deux services

**Fichier à montrer :** [docker-compose.yml](docker-compose.yml)

Docker Compose orchestre **deux services simultanément** :

```
Service 1 : api
  - Image : construite depuis le Dockerfile local
  - Port  : 8000 → 8000
  - Volumes : ./models et ./data/processed montés (le modèle est partagé entre host et conteneur)
  - Restart : unless-stopped (redémarre automatiquement si crash)

Service 2 : mlflow
  - Image : python avec mlflow installé
  - Port  : 5000 → 5000
  - Volume : ./mlruns monté (persistance des expériences)
  - Command : mlflow ui --host 0.0.0.0 --port 5000
```

**Commande à montrer :**
```bash
docker-compose up --build
# → Lance l'API sur localhost:8000
# → Lance MLflow UI sur localhost:5000
```

---

## PARTIE 6 — L'Automatisation CI/CD avec GitHub Actions (4 min)

### 6.1 Le pipeline principal

**Fichier à montrer :** [.github/workflows/ml_pipeline.yml](.github/workflows/ml_pipeline.yml)

Ce workflow est déclenché **automatiquement** à chaque push sur `main` ou `master`, et à chaque Pull Request. Il exécute dans l'ordre :

```
1. Checkout du code
2. Installation de Python 3.10 + uv
3. Installation des dépendances (uv pip install)
4. Téléchargement du dataset depuis Kaggle (secrets: KAGGLE_USERNAME, KAGGLE_KEY)
5. Prétraitement (preprocess.py)
6. Entraînement (train.py — GridSearchCV)
7. Évaluation (evaluate.py — métriques sur test set)
8. Upload des artefacts : evaluation_plots.png + best_model.pkl
```

**Point clé :** Le dataset n'est pas stocké dans le repo Git (trop lourd). Il est téléchargé à la volée depuis Kaggle grâce aux secrets GitHub.

### 6.2 Le workflow de ré-entraînement planifié

**Fichier à montrer :** [.github/workflows/retrain.yml](.github/workflows/retrain.yml)

Ce workflow se déclenche de **deux façons** :
- **Automatiquement** : tous les lundis à 2h du matin UTC (cron : `"0 2 * * 1"`)
- **Manuellement** : via `workflow_dispatch` avec choix du batch (`val` ou `test`)

Cela simule un **système de ré-entraînement continu** : chaque semaine, de nouvelles données sont intégrées et le modèle est mis à jour automatiquement.

### 6.3 Les secrets GitHub nécessaires

| Secret | Utilisation |
|---|---|
| `KAGGLE_USERNAME` | Authentification Kaggle |
| `KAGGLE_KEY` | Authentification Kaggle |
| `AZURE_STORAGE_CONNECTION_STRING` | Upload Azure (optionnel) |
| `AZURE_CONTAINER_NAME` | Container Azure (optionnel) |

**Fichier à montrer :** [.env.example](.env.example) — montrer les variables d'environnement nécessaires.

---

## PARTIE 7 — Orchestration Locale (2 min)

### 7.1 Le Makefile

**Fichier à montrer :** [Makefile](Makefile)

Le Makefile expose des commandes simples pour chaque étape du pipeline :

```bash
make preprocess    # Lancer le prétraitement
make train         # Lancer l'entraînement
make evaluate      # Lancer l'évaluation
make retrain       # Lancer le ré-entraînement
make serve         # Démarrer l'API sur :8000
make mlflow-ui     # Démarrer MLflow UI sur :5000
make run-all       # Pipeline complet
```

### 7.2 Le script de pipeline complet

**Fichier à montrer :** [run_pipeline.sh](run_pipeline.sh)

Ce script Bash exécute le pipeline de bout en bout en une seule commande :

```bash
bash run_pipeline.sh
# → preprocess → train → evaluate → retrain(val) → retrain(test)
```

Une version PowerShell est également disponible pour Windows : [run_pipeline.ps1](run_pipeline.ps1)

---

## PARTIE 8 — Le Package Python Installable (1 min)

**Fichier à montrer :** [setup.py](setup.py)

Le projet est structuré comme un **vrai package Python installable** via pip :

```bash
pip install -e .
# ou avec uv :
uv pip install -e .
```

Cela permet d'importer les modules depuis n'importe où :
```python
from spotify_mlops.preprocess import preprocess_data
from spotify_mlops.train import train_model
```

C'est une pratique professionnelle qui distingue un script jetable d'un vrai projet maintenable.

---

## CONCLUSION (2 min)

### Récapitulatif des composants

| Composant | Technologie | Rôle |
|---|---|---|
| Données | Kaggle + CSV | Source de données brutes |
| Preprocessing | scikit-learn | Nettoyage, encodage, split |
| Modélisation | scikit-learn GridSearchCV | Sélection automatique du meilleur modèle |
| Tracking | MLflow | Traçabilité de toutes les expériences |
| API | FastAPI + Uvicorn | Mise en production du modèle |
| Conteneurisation | Docker + Compose | Déploiement reproductible |
| CI/CD | GitHub Actions | Automatisation du pipeline complet |
| Stockage cloud | Azure Blob Storage | Persistance optionnelle des données |

### Ce que démontre ce projet

1. **Reproductibilité** : random_state=42, parquet, preprocesseur sauvegardé
2. **Pas de data leakage** : preprocesseur fitté uniquement sur le train
3. **Automatisation** : GitHub Actions déclenche le pipeline sans intervention humaine
4. **Traçabilité** : chaque run MLflow est un snapshot complet (params + metrics + modèle)
5. **Maintenabilité** : code structuré en package Python avec Makefile et scripts
6. **Opérabilité** : API avec health check, hot-reload, logging des requêtes
7. **Scalabilité** : Docker permet de déployer sur n'importe quelle infrastructure

### Chiffres clés à retenir

- **8 000** enregistrements dans le dataset
- **9** features prédictives (5 numériques + 4 catégorielles)
- **3** modèles comparés (LinearRegression, Ridge, Lasso)
- **R² = 0.7726** sur le test set (77% de variance expliquée)
- **MAE = 3.02** publicités/semaine d'erreur moyenne
- **2** services Docker (API + MLflow)
- **2** workflows GitHub Actions (pipeline + ré-entraînement hebdomadaire)

---

## ORDRE DE NAVIGATION DES FICHIERS PENDANT L'ORAL

```
1.  README.md                              ← Introduction générale
2.  data/raw/spotify_churn_dataset.csv     ← Les données sources
3.  notebooks/01_eda.ipynb                 ← Exploration des données
4.  src/preprocess.py                      ← Étape 1 : Prétraitement
5.  data/processed/                        ← Résultats du prétraitement
6.  src/train.py                           ← Étape 2 : Entraînement
7.  models/best_model.pkl                  ← Modèle sauvegardé
8.  src/evaluate.py                        ← Étape 3 : Évaluation
9.  reports/evaluation_plots.png           ← Graphiques de performance
10. http://localhost:5000                  ← MLflow UI (live)
11. src/retrain.py                         ← Ré-entraînement
12. src/serve.py                           ← API FastAPI
13. http://localhost:8000/docs             ← Swagger UI (live)
14. Dockerfile                             ← Conteneurisation
15. docker-compose.yml                     ← Orchestration Docker
16. .github/workflows/ml_pipeline.yml      ← CI/CD principal
17. .github/workflows/retrain.yml          ← CI/CD planifié
18. Makefile                               ← Commandes locales
19. setup.py                               ← Package Python
```

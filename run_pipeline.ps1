# Pipeline MLOps complet -- Spotify Ads Prediction
# Usage : .\run_pipeline.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($msg) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
}

# 0. Venv
Step "0. Venv"
if (-not (Test-Path ".venv")) {
    Write-Host "Creation du venv..."
    uv venv
}
& ".venv\Scripts\Activate.ps1"
Write-Host "Python : $(python --version)"

# 1. Dependances
Step "1. Installation des dependances"
uv pip install -r requirements.txt
pip install -e . --quiet
Write-Host "Package spotify_mlops installe."

# 2. Dataset Kaggle
Step "2. Telechargement du dataset Kaggle"
if (Test-Path "data\raw\spotify_churn_dataset.csv") {
    Write-Host "Dataset deja present -- telechargement ignore." -ForegroundColor Yellow
} else {
    if (-not (Test-Path "data\raw")) {
        New-Item -ItemType Directory -Path "data\raw" | Out-Null
    }
    kaggle datasets download -d nabihazahid/spotify-dataset-for-churn-analysis -p data/raw/ --unzip
    Write-Host "Dataset telecharge dans data\raw\"
}

# 3. Pretraitement
Step "3. Pretraitement (split 70/15/15 + encoding)"
python src/preprocess.py

# 4. Entrainement
Step "4. Entrainement (GridSearchCV + MLflow)"
python src/train.py

# 5. Evaluation
Step "5. Evaluation finale (test set + plots)"
python src/evaluate.py
Write-Host "Plots -> reports\evaluation_plots.png"

# 6. Reentrainement batch val
Step "6. Reentrainement -- batch val"
python src/retrain.py --batch val

# 7. Reentrainement batch test
Step "7. Reentrainement -- batch test"
python src/retrain.py --batch test

# Resume
Write-Host ""
Write-Host "Pipeline termine avec succes !" -ForegroundColor Green
Write-Host ""
Write-Host "Prochaines etapes :" -ForegroundColor Cyan
Write-Host "  mlflow ui                       -> http://localhost:5000"
Write-Host "  uvicorn src.serve:app --reload  -> http://localhost:8000/docs"
Write-Host "  Invoke-RestMethod -Method Post http://localhost:8000/reload"
Write-Host ""
Write-Host "Ou via Docker :"
Write-Host "  docker-compose up --build"
Write-Host ""

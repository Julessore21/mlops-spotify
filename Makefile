install:
	uv pip install -r requirements.txt

install-dev:
	uv pip install -e .

preprocess:
	python src/preprocess.py

train:
	python src/train.py

evaluate:
	python src/evaluate.py

retrain-val:
	python src/retrain.py --batch val

retrain-test:
	python src/retrain.py --batch test

run-all: preprocess train evaluate

serve:
	uvicorn src.serve:app --reload

mlflow-ui:
	mlflow ui

demo:
	@echo "=== 1. Preprocessing (70/15/15) ==="
	python src/preprocess.py
	@echo "\n=== 2. Entrainement initial (GridSearch + MLflow) ==="
	python src/train.py
	@echo "\n=== 3. Evaluation ==="
	python src/evaluate.py
	@echo "\n=== 4. Retrain batch val (nouvelles donnees) ==="
	python src/retrain.py --batch val
	@echo "\n=== 5. Retrain batch test (nouvelles donnees) ==="
	python src/retrain.py --batch test
	@echo "\n=== 6. Notebook dataviz (MLOps Story) ==="
	jupyter nbconvert --to notebook --execute notebooks/03_mlops_story.ipynb --output notebooks/03_mlops_story_executed.ipynb
	@echo "\n=== DONE — Lance: make mlflow-ui et make serve ==="

clean:
	rm -rf data/processed models reports mlruns

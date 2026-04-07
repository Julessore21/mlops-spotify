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

clean:
	rm -rf data/processed models reports mlruns

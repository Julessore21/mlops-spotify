install:
	pip install -r requirements.txt

preprocess:
	python src/preprocess.py

train:
	python src/train.py

evaluate:
	python src/evaluate.py

run-all: preprocess train evaluate

mlflow-ui:
	mlflow ui

clean:
	rm -rf data/processed models reports mlruns

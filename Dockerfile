FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY setup.py .
COPY src/ ./src/
COPY data/processed/preprocessor.pkl ./data/processed/preprocessor.pkl
COPY models/best_model.pkl ./models/best_model.pkl

EXPOSE 8000

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.10-slim-bookworm

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY setup.py .
COPY src/ ./src/
COPY data/processed/preprocessor.pkl ./data/processed/preprocessor.pkl
COPY models/best_model.pkl ./models/best_model.pkl

EXPOSE 8000

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]

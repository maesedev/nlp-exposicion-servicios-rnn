FROM python:3.12-slim

WORKDIR /app

# Dependencies first (layer cache)
COPY ["008 web/app/requirements.txt", "requirements.txt"]
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY ["008 web/app/main.py", "main.py"]

# BPE tokenizer (imported directly, no external path needed)
COPY ["001 Proprocesamiento/token_encoder.py", "token_encoder.py"]

# BPE assets
COPY ["001 Proprocesamiento/tokens/vocabulary.txt", "assets/vocabulary.txt"]
COPY ["001 Proprocesamiento/tokens/merges.csv",     "assets/merges.csv"]

ENV MLFLOW_TRACKING_URI="http://localhost:5000"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

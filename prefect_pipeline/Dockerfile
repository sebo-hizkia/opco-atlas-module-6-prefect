FROM python:3.11-slim

WORKDIR /app

# Dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY flow.py .

# Variables utiles (peuvent être surchargées par docker-compose.yml)
ENV SERVE_INTERVAL_SECONDS=10
ENV DRIFT_THRESHOLD=0.5

CMD ["python", "flow.py"]

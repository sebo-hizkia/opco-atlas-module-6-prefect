FROM python:3.11-slim

WORKDIR /app

# Dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY flow.py .

CMD ["python", "flow.py"]

FROM python:3.10-slim

WORKDIR /app

# Sakinisha mfumo wa kuendesha ONNX runtime na Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Back4App inatumia Port 8080 kama default container port
EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]

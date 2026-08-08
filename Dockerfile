# Clean, Railway-friendly image (the example's uv/pipecat-base Dockerfile doesn't build here).
FROM python:3.12-slim

WORKDIR /app

# build-essential covers any package that needs to compile a wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# server.py binds 0.0.0.0:$PORT (Railway injects PORT) and serves /dialout, /twiml, /ws.
CMD ["python", "server.py"]

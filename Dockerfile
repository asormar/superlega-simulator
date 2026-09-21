# Single-container demo: FastAPI serves the API and the built React frontend.
# Target: Render (Docker runtime); the platform injects $PORT (7860 locally).

# ── Stage 1: build the frontend ──────────────────────────────
FROM node:22-slim AS web
WORKDIR /web
COPY src/web/package.json src/web/package-lock.json ./
RUN npm ci
COPY src/web/ ./
RUN npm run build

# ── Stage 2: Python runtime ──────────────────────────────────
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860
WORKDIR /home/user/app

COPY --chown=user requirements-deploy.txt ./
RUN pip install --no-cache-dir --user -r requirements-deploy.txt

COPY --chown=user sitecustomize.py ./
COPY --chown=user src/ ./src/
COPY --chown=user DB/ ./DB/
COPY --chown=user models/ ./models/
COPY --chown=user --from=web /web/dist ./src/web/dist

EXPOSE 7860
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]

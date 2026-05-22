# Deployment Guide

This guide covers local development, demo sharing, Docker, and cloud deployment.

## Local Development

```bash
copy .env.example .env
pip install -r requirements.txt
python ingest.py
python api.py
```

Open `http://localhost:8000`.

Required environment variable:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

Recommended for deployed environments:

```env
RAG_API_KEY=your_private_app_api_key
```

## Quick Demo Sharing

### Ngrok

1. Start the app with `python api.py`.
2. In another terminal, run `ngrok http 8000`.
3. Share the HTTPS URL that ngrok provides.

### Railway

1. Push this repo to GitHub.
2. Create a Railway project from the GitHub repository.
3. Add `GOOGLE_API_KEY` and `RAG_API_KEY` in Railway environment variables.
4. Deploy with the included `Dockerfile`.

## Docker

```bash
docker compose up --build
```

The container serves the API on `http://localhost:8000`.

Persistent local mounts:

- `./data:/app/data`
- `./qdrant_store:/app/qdrant_store`

## Cloud VM

1. Create a VM.
2. Install Docker and Docker Compose.
3. Clone this repository.
4. Create `.env`.
5. Run:

```bash
docker compose up -d --build
```

For production, put the app behind a reverse proxy such as nginx or Caddy and terminate HTTPS at the proxy.

## Pre-Demo Checklist

- `.env` has a valid `GOOGLE_API_KEY`.
- `RAG_API_KEY` is set if the app is public.
- Documents are placed in `data/`.
- Ingestion has completed successfully.
- `GET /health` returns `{"status": "ok", "initialized": true}`.
- The UI can upload, ingest, create sessions, and answer a test question.


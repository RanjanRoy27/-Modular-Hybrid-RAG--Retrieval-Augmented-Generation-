# RAG MVP — Deployment Guide

This guide covers how to run your RAG application locally, share it as an MVP demo, and deploy it to a cloud provider.

---

## 1. Running Locally (Development)

```bash
# Make sure your .env file has your key:
# GOOGLE_API_KEY=your_key_here

cd rag-mvp

# Install dependencies
pip install -r requirements.txt

# Ingest your documents first
python ingest.py

# Start the server
python api.py
```

Then open **[http://localhost:8000](http://localhost:8000)** in your browser to see the chat UI.

---

## 2. Sharing as an MVP Demo (Quickest Path)

### Option A — Ngrok (Zero Deployment, Share a Public URL in 2 Minutes)

1. Download ngrok from [https://ngrok.com/download](https://ngrok.com/download)  
2. Start your server: `python api.py`  
3. In a new terminal: `ngrok http 8000`  
4. Ngrok gives you a public URL like `https://abc123.ngrok.io` — share this with anyone!

### Option B — Railway (Free Tier, Permanent URL)

1. Push your code to a GitHub repo.
2. Sign up at [https://railway.app](https://railway.app).
3. Click **New Project → Deploy from GitHub Repo**.
4. Add your environment variable: `GOOGLE_API_KEY` in the Railway dashboard.
5. Railway auto-detects the `Dockerfile` and deploys. Your app gets a free `*.railway.app` URL.

---

## 3. Docker Deployment

### Build and Run Locally

```bash
# Add your API key to .env first

# Build the image
docker build -t rag-mvp .

# Run with docker-compose (recommended — mounts data/ and vector_store/)
docker-compose up -d

# Check it's running
curl http://localhost:8000/health
```

### Deploy to a Cloud VM (GCP, AWS, DigitalOcean)

1. **Create a VM** (e.g. GCP `e2-small`, AWS `t3.micro`, DigitalOcean Droplet — all have free tiers or ~$6/mo).
2. **SSH into the VM** and install Docker:
   ```bash
   apt-get update && apt-get install -y docker.io docker-compose
   ```
3. **Clone or copy your project** to the VM.
4. **Create the `.env` file** with your `GOOGLE_API_KEY`.
5. **Run:**
   ```bash
   docker-compose up -d
   ```
6. Open the VM's public IP on port `8000` in your browser.

**Optional:** Point a domain (e.g. from Namecheap) to the VM's IP and add an nginx reverse proxy + SSL with Certbot (free HTTPS).

---

## 4. Recommended MVP Architecture

```
Customer Browser
      │
      ▼
[your-domain.com:443]  ← nginx (SSL/TLS termination)
      │
      ▼
[localhost:8000]  ← FastAPI + RAG (Docker container)
      │
      ├── FAISS Vector Store (local file)
      ├── BM25 Index (local file)
      └── Gemini API (Google Cloud)
```

---

## 5. Pre-Demo Checklist

- [ ] `.env` file has a valid `GOOGLE_API_KEY`
- [ ] Documents placed in `data/` folder
- [ ] `python ingest.py` (or `/rag/ingest` API call) completed successfully
- [ ] `GET /health` returns `{"initialized": true}`
- [ ] Test a few questions in the UI before showing customers
- [ ] If sharing via ngrok, start ngrok *after* your server is running

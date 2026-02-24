# 🧠 Hybrid RAG MVP (FastAPI + Google Gemini)

A production-ready **Retrieval-Augmented Generation (RAG)** system built with Python, FastAPI, and the Google Gemini ecosystem. This project allows you to have natural, persistent conversations with your own documents (PDF, DOCX, TXT) using a state-of-the-art hybrid search engine.

---

## 🔥 Key Features

- **Hybrid Search Engine**: Combines **BM25 (Keyword)** and **FAISS (Semantic)** searching using LangChain's `EnsembleRetriever` for superior accuracy.
- **Multi-Format Ingestion**: Automatically detects and processes `.pdf`, `.docx`, and `.txt` files.
- **Persistent Conversational Memory**: Remembers chat history across restarts using a JSON-based session storage.
- **FastAPI Backend**: Production-ready API for easy integration with web or mobile frontends.
- **Contextual Rephrasing**: Follow-up questions are automatically rephrased into standalone search queries for better retrieval.
- **Performance Logging**: Detailed tracking of query latency, context length, and model usage.

---

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **LLM / Embeddings**: Google Gemini (Flash 1.5, Embedding-001)
- **Orchestration**: LangChain
- **Vector Store**: FAISS
- **Search Logic**: BM25 & Semantic Ensemble
- **Document Loading**: PyPDF, Docx2txt

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- A Google Gemini API Key ([Get it here](https://aistudio.google.com/))

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/rag-mvp.git
cd rag-mvp

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_api_key_here
```

### 4. Usage

#### Step A: Ingest Documents
Place your files in the `data/` folder and run:
```bash
python ingest.py
```

#### Step B: Start the API
```bash
python api.py
```
The API will be available at `http://localhost:8000`. You can view the Interactive Documentation at `http://localhost:8000/docs`.

---

## 📡 API Endpoints

### `POST /rag/ingest`
Triggers the ingestion pipeline to refresh the vector store from the `data/` directory.

### `POST /rag/answer`
Answers a question based on uploaded documents. Supports chat history.
**Payload:**
```json
{
  "question": "What is the third layer?",
  "history": [
    {"role": "human", "content": "Tell me about the architecture."},
    {"role": "ai", "content": "..."}
  ]
}
```

---

## 📂 Project Structure
```text
├── api.py           # FastAPI service wrapper
├── ingest.py        # Document ingestion & indexing logic
├── main.py          # CLI-based chatbot version
├── test_api.py      # E2E API verification script
├── requirements.txt # Project dependencies
└── data/            # Your source documents (.pdf, .docx, .txt)
```

---

## 📝 License
This project is licensed under the MIT License.

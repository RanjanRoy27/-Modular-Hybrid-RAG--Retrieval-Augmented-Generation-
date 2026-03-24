import os
import time
import json
import uuid
import shutil
import logging
import asyncio
from datetime import datetime
from typing import List, Optional, AsyncIterator
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import core
import ingest
import monitor
import agent
from langchain_core.messages import HumanMessage, AIMessage
from reranker import CrossEncoderReranker

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("rag_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RAG-API")

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SESSIONS_FILE = os.path.join(os.path.dirname(__file__), "sessions.json")
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# ── App ─────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG MVP API",
    description="FastAPI wrapper for the Google Generative AI RAG system.",
    version="2.0.0"
)

# Serve static UI files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Auth ───────────────────────────────────────────────────────────────────────
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    expected_key = os.getenv("RAG_API_KEY", "default-dev-key")
    if credentials.credentials != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return credentials

# ── RAG State ──────────────────────────────────────────────────────────────────
class RAGState:
    def __init__(self):
        self.agent_executor = None
        self.initialized = False

    def initialize(self):
        if not core.validate_env():
            logger.error("API start failed: GOOGLE_API_KEY missing.")
            return
        logger.info("Initializing Agent components...")
        self.agent_executor = agent.build_agent()
        self.initialized = True
        logger.info("Agent components ready.")

state = RAGState()

@app.on_event("startup")
async def startup_event():
    os.makedirs(DATA_DIR, exist_ok=True)
    state.initialize()

# ── Session Storage ────────────────────────────────────────────────────────────
def load_sessions() -> dict:
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"sessions": [], "active_session_id": None}

def save_sessions(data: dict):
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_data_files() -> List[str]:
    """Returns list of supported files currently in the data/ directory."""
    if not os.path.exists(DATA_DIR):
        return []
    return sorted([
        f for f in os.listdir(DATA_DIR)
        if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS
    ])

# ── Request / Response Models ───────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    history: List[dict] = []
    session_id: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str
    latency_ms: float
    context_length: int
    model: str
    citations: List[dict] = []

class IngestResponse(BaseModel):
    status: str
    files_loaded: int
    chunks_created: int
    message: str
    session_id: str

class SessionCreate(BaseModel):
    name: Optional[str] = None

class MessageAppend(BaseModel):
    messages: List[dict]  # [{role, content}]

# ── Helper ─────────────────────────────────────────────────────────────────────
def build_chat_history(history: List[dict]):
    result = []
    for msg in history:
        if msg.get("role") == "human":
            result.append(HumanMessage(content=msg["content"]))
        else:
            result.append(AIMessage(content=msg["content"]))
    return result

# ── Routes ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "RAG MVP API — docs at /docs"}


@app.get("/health")
async def health():
    return {"status": "ok", "initialized": state.initialized}


# ── File Management ─────────────────────────────────────────────────────────────

@app.get("/rag/files", dependencies=[Depends(verify_token)])
async def list_files():
    """List all document files currently in the data/ directory."""
    files = get_data_files()
    file_details = []
    for f in files:
        path = os.path.join(DATA_DIR, f)
        stat = os.stat(path)
        file_details.append({
            "name": f,
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": os.path.splitext(f)[1].lower()
        })
    return {"files": file_details, "count": len(file_details)}


@app.post("/rag/upload", dependencies=[Depends(verify_token)])
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload one or more documents to the data/ directory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    saved = []
    errors = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f"{file.filename}: unsupported format (use PDF, DOCX, or TXT)")
            continue
        dest = os.path.join(DATA_DIR, file.filename)
        try:
            with open(dest, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved.append(file.filename)
            logger.info(f"Uploaded: {file.filename}")
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")

    return {
        "saved": saved,
        "errors": errors,
        "total_files": len(get_data_files())
    }


@app.delete("/rag/files/{filename}", dependencies=[Depends(verify_token)])
async def delete_file(filename: str):
    """Remove a file from the data/ directory."""
    # Sanitize — no path traversal
    safe_name = os.path.basename(filename)
    path = os.path.join(DATA_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    os.remove(path)
    logger.info(f"Deleted: {safe_name}")
    return {"deleted": safe_name, "total_files": len(get_data_files())}


# ── Ingestion ──────────────────────────────────────────────────────────────────

@app.post("/rag/ingest", response_model=IngestResponse, dependencies=[Depends(verify_token)])
async def ingest_endpoint():
    """Trigger document re-ingestion and create a new chat session."""
    logger.info("Ingestion requested via API.")
    res = ingest.ingest_docs()

    if not res["success"]:
        raise HTTPException(status_code=500, detail=res["error"])

    state.initialize()  # Reload retriever

    # Auto-create a new session linked to current files
    current_files = get_data_files()
    data = load_sessions()

    ts = datetime.now().strftime("%b %d, %H:%M")
    file_names = [os.path.splitext(f)[0] for f in current_files[:2]]
    session_name = f"{', '.join(file_names)}" if file_names else f"Session {ts}"
    if len(current_files) > 2:
        session_name += f" +{len(current_files) - 2} more"

    new_session = {
        "id": str(uuid.uuid4()),
        "name": session_name,
        "created_at": datetime.now().isoformat(),
        "files": current_files,
        "messages": []
    }
    data["sessions"].insert(0, new_session)
    data["active_session_id"] = new_session["id"]
    save_sessions(data)

    return {
        "status": "success",
        "files_loaded": res["files_loaded"],
        "chunks_created": res["chunks_created"],
        "message": f"Ingested {res['files_loaded']} files, {res['chunks_created']} chunks.",
        "session_id": new_session["id"]
    }


# ── Session Management ─────────────────────────────────────────────────────────

@app.get("/rag/sessions", dependencies=[Depends(verify_token)])
async def list_sessions():
    """Get all saved chat sessions."""
    data = load_sessions()
    return {
        "sessions": data.get("sessions", []),
        "active_session_id": data.get("active_session_id")
    }


@app.post("/rag/sessions", dependencies=[Depends(verify_token)])
async def create_session(body: SessionCreate):
    """Create a new blank chat session."""
    data = load_sessions()
    current_files = get_data_files()

    ts = datetime.now().strftime("%b %d, %H:%M")
    new_session = {
        "id": str(uuid.uuid4()),
        "name": body.name or f"New Chat — {ts}",
        "created_at": datetime.now().isoformat(),
        "files": current_files,
        "messages": []
    }
    data["sessions"].insert(0, new_session)
    data["active_session_id"] = new_session["id"]
    save_sessions(data)
    return new_session


@app.put("/rag/sessions/{session_id}", dependencies=[Depends(verify_token)])
async def update_session(session_id: str, body: dict):
    """Update session name."""
    data = load_sessions()
    for s in data["sessions"]:
        if s["id"] == session_id:
            if "name" in body:
                s["name"] = body["name"]
            save_sessions(data)
            return s
    raise HTTPException(status_code=404, detail="Session not found")


@app.put("/rag/sessions/{session_id}/messages", dependencies=[Depends(verify_token)])
async def save_session_messages(session_id: str, body: MessageAppend):
    """Persist messages to a session (called from frontend after each exchange)."""
    data = load_sessions()
    for s in data["sessions"]:
        if s["id"] == session_id:
            s["messages"] = body.messages
            
            # Simple flagging for episodic memory summarization (Priority 5 prep)
            if len(s["messages"]) >= 20 and not s.get("summarized"):
                logger.info(f"Session {session_id} exceeded 10 turns. Flagging for summarization.")
                s["summarized"] = True
                
            save_sessions(data)
            return {"ok": True}
    raise HTTPException(status_code=404, detail="Session not found")


@app.put("/rag/sessions/active/{session_id}", dependencies=[Depends(verify_token)])
async def set_active_session(session_id: str):
    """Mark a session as active."""
    data = load_sessions()
    ids = [s["id"] for s in data["sessions"]]
    if session_id not in ids:
        raise HTTPException(status_code=404, detail="Session not found")
    data["active_session_id"] = session_id
    save_sessions(data)
    return {"active_session_id": session_id}


@app.delete("/rag/sessions/{session_id}", dependencies=[Depends(verify_token)])
async def delete_session(session_id: str):
    """Delete a chat session."""
    data = load_sessions()
    before = len(data["sessions"])
    data["sessions"] = [s for s in data["sessions"] if s["id"] != session_id]
    if len(data["sessions"]) == before:
        raise HTTPException(status_code=404, detail="Session not found")
    if data.get("active_session_id") == session_id:
        data["active_session_id"] = data["sessions"][0]["id"] if data["sessions"] else None
    save_sessions(data)
    return {"deleted": session_id}


# ── Answer Endpoints ───────────────────────────────────────────────────────────

@app.post("/rag/answer", response_model=AnswerResponse, dependencies=[Depends(verify_token)])
async def answer_endpoint(request: QuestionRequest):
    """Standard (non-streaming) answer endpoint."""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")

    start_time = time.time()
    query_id = str(uuid.uuid4())
    logger.info(f"Question: {request.question}")

    try:
        chat_history = build_chat_history(request.history)

        # Execute through Agent
        response = state.agent_executor.invoke({
            "input": request.question,
            "chat_history": chat_history
        })
        
        output_data = response.get("output", "")
        
        # Parse strict JSON
        try:
            import json
            # Handle potential markdown code blocks returned by LLM
            clean_text = output_data.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_text)
            answer = parsed.get("answer", output_data)
            citations = parsed.get("citations", [])
        except Exception:
            answer = output_data
            citations = []

        total_latency = (time.time() - start_time) * 1000

        monitor.log_query(
            query_id=query_id,
            question=request.question,
            chunks_retrieved=len(citations), # Approximated via citations
            retrieved_texts=[str(c) for c in citations],
            rerank_scores=[],
            llm_latency_ms=total_latency * 0.9,
            total_latency_ms=total_latency,
            fallback_triggered=False
        )

        return {
            "answer": answer, 
            "latency_ms": round(total_latency, 2),
            "context_length": len(citations), 
            "model": "gemini-flash-agent",
            "citations": citations
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/stream", dependencies=[Depends(verify_token)])
async def stream_endpoint(request: QuestionRequest):
    """Streaming answer via Server-Sent Events."""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")

    async def event_generator() -> AsyncIterator[str]:
        start_time = time.time()
        try:
            chat_history = build_chat_history(request.history)

            search_query = request.question
            if chat_history:
                rephrase_res = state.rephrase_chain.invoke({
                    "chat_history": chat_history, "question": request.question
                })
                search_query = core.clean_ai_content(rephrase_res.content)

            initial_docs = state.retriever.invoke(search_query)
            docs = state.reranker.rerank(search_query, initial_docs, top_k=5)
            context = "\n\n".join([doc.page_content for doc in docs])
            context_len = len(context)

            # Stream document sources in a first event
            sources = list(set(doc.metadata.get("source", "") for doc in docs if doc.metadata.get("source")))
            if sources:
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            full_answer = ""
            async for chunk in state.qa_chain.astream({
                "context": context, "chat_history": chat_history, "question": request.question
            }):
                token = core.clean_ai_content(chunk.content)
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                    await asyncio.sleep(0)

            latency = (time.time() - start_time) * 1000
            yield f"data: {json.dumps({'type': 'metadata', 'latency_ms': round(latency, 2), 'context_length': context_len, 'model': 'gemini-flash-latest'})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"[STREAM] Error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

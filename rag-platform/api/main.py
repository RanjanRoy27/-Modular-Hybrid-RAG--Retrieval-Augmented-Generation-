import asyncio
import json
import logging
import os
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, List, Optional

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLATFORM_ROOT.parent
for path in (str(PLATFORM_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage

import agent
import core
import ingest
import monitor
from core.reranker import CrossEncoderReranker
from domains.accounting.preprocessor import AccountingDomain
from domains.base_domain import BaseDomain
from domains.bookkeeping.preprocessor import BookkeepingDomain
from domains.legal.preprocessor import LegalDomain

try:
    from .schemas import AnswerResponse, IngestResponse, MessageAppend, QuestionRequest, SessionCreate
except ImportError:
    from schemas import AnswerResponse, IngestResponse, MessageAppend, QuestionRequest, SessionCreate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("rag_api.log"), logging.StreamHandler()],
)
logger = logging.getLogger("RAG-API")

DATA_DIR = REPO_ROOT / "data"
SESSIONS_FILE = REPO_ROOT / "sessions.json"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

DOMAIN_REGISTRY = {
    "accounting": AccountingDomain,
    "legal": LegalDomain,
    "bookkeeping": BookkeepingDomain,
}


def get_domain(domain: str) -> BaseDomain:
    if domain not in DOMAIN_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown domain: {domain}")
    return DOMAIN_REGISTRY[domain]()


app = FastAPI(
    title="RAG MVP API",
    description="FastAPI wrapper for the Google Generative AI RAG system.",
    version="2.0.0",
)

STATIC_DIR = REPO_ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

security = HTTPBearer(auto_error=False)


def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    expected_key = os.getenv("RAG_API_KEY")
    if not expected_key:
        return credentials
    if not credentials or credentials.credentials != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return credentials


class RAGState:
    def __init__(self):
        self.agent_executor = None
        self.retriever = None
        self.rephrase_chain = None
        self.qa_chain = None
        self.reranker = None
        self.initialized = False

    def initialize(self, domain_obj: BaseDomain | None = None):
        domain_obj = domain_obj or get_domain("accounting")
        config = domain_obj.get_config()
        if not core.validate_env():
            logger.error("API start failed: GOOGLE_API_KEY missing.")
            return
        logger.info("Initializing RAG components for domain=%s...", config.get("domain"))
        embeddings = core.get_embeddings()
        llm = core.get_llm()
        retriever, msg = core.load_retriever(
            embeddings,
            top_k=int(config.get("top_k", 5)),
            bm25_weight=config.get("bm25_weight"),
            semantic_weight=config.get("semantic_weight"),
        )
        if not retriever:
            logger.error(msg)
            self.initialized = False
            return
        self.retriever = retriever
        self.rephrase_chain = core.rephrase_prompt_template() | llm
        self.qa_chain = core.qa_prompt_template(domain_obj.get_system_prompt()) | llm
        self.reranker = CrossEncoderReranker()
        self.agent_executor = agent.build_agent(domain_obj)
        self.initialized = True
        logger.info("RAG components ready.")


state = RAGState()


@app.on_event("startup")
async def startup_event():
    DATA_DIR.mkdir(exist_ok=True)
    state.initialize()


def load_sessions() -> dict:
    if SESSIONS_FILE.exists():
        try:
            with SESSIONS_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"sessions": [], "active_session_id": None}


def save_sessions(data: dict):
    with SESSIONS_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_data_files() -> List[str]:
    if not DATA_DIR.exists():
        return []
    return sorted([
        f.name for f in DATA_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
    ])


def build_chat_history(history: List[dict]):
    result = []
    for msg in history:
        if msg.get("role") == "human":
            result.append(HumanMessage(content=msg["content"]))
        else:
            result.append(AIMessage(content=msg["content"]))
    return result


@app.get("/", include_in_schema=False)
async def serve_ui():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "RAG MVP API - docs at /docs"}


@app.get("/health")
async def health():
    return {"status": "ok", "initialized": state.initialized}


@app.get("/rag/files", dependencies=[Depends(verify_token)])
async def list_files():
    files = get_data_files()
    file_details = []
    for f in files:
        path = DATA_DIR / f
        stat = path.stat()
        file_details.append({
            "name": f,
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": path.suffix.lower(),
        })
    return {"files": file_details, "count": len(file_details)}


@app.post("/rag/upload", dependencies=[Depends(verify_token)])
async def upload_files(files: List[UploadFile] = File(...)):
    DATA_DIR.mkdir(exist_ok=True)
    saved = []
    errors = []
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f"{file.filename}: unsupported format (use PDF, DOCX, or TXT)")
            continue
        dest = DATA_DIR / file.filename
        try:
            with dest.open("wb") as f:
                shutil.copyfileobj(file.file, f)
            saved.append(file.filename)
            logger.info("Uploaded: %s", file.filename)
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")
    return {"saved": saved, "errors": errors, "total_files": len(get_data_files())}


@app.delete("/rag/files/{filename}", dependencies=[Depends(verify_token)])
async def delete_file(filename: str):
    safe_name = os.path.basename(filename)
    path = DATA_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    path.unlink()
    logger.info("Deleted: %s", safe_name)
    return {"deleted": safe_name, "total_files": len(get_data_files())}


@app.post("/rag/ingest", response_model=IngestResponse, dependencies=[Depends(verify_token)])
async def ingest_endpoint(domain: str = "accounting"):
    domain_obj = get_domain(domain)
    logger.info("Ingestion requested via API for domain=%s.", domain)
    res = ingest.ingest_docs(domain_obj)
    if not res["success"]:
        raise HTTPException(status_code=500, detail=res["error"])
    state.initialize(domain_obj)

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
        "messages": [],
    }
    data["sessions"].insert(0, new_session)
    data["active_session_id"] = new_session["id"]
    save_sessions(data)
    return {
        "status": "success",
        "files_loaded": res["files_loaded"],
        "chunks_created": res["chunks_created"],
        "message": f"Ingested {res['files_loaded']} files, {res['chunks_created']} chunks.",
        "session_id": new_session["id"],
    }


@app.get("/rag/sessions", dependencies=[Depends(verify_token)])
async def list_sessions():
    data = load_sessions()
    return {"sessions": data.get("sessions", []), "active_session_id": data.get("active_session_id")}


@app.post("/rag/sessions", dependencies=[Depends(verify_token)])
async def create_session(body: SessionCreate):
    data = load_sessions()
    current_files = get_data_files()
    ts = datetime.now().strftime("%b %d, %H:%M")
    new_session = {
        "id": str(uuid.uuid4()),
        "name": body.name or f"New Chat - {ts}",
        "created_at": datetime.now().isoformat(),
        "files": current_files,
        "messages": [],
    }
    data["sessions"].insert(0, new_session)
    data["active_session_id"] = new_session["id"]
    save_sessions(data)
    return new_session


@app.put("/rag/sessions/{session_id}", dependencies=[Depends(verify_token)])
async def update_session(session_id: str, body: dict):
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
    data = load_sessions()
    for s in data["sessions"]:
        if s["id"] == session_id:
            s["messages"] = body.messages
            if len(s["messages"]) >= 20 and not s.get("summarized"):
                logger.info("Session %s exceeded 10 turns. Flagging for summarization.", session_id)
                s["summarized"] = True
            save_sessions(data)
            return {"ok": True}
    raise HTTPException(status_code=404, detail="Session not found")


@app.put("/rag/sessions/active/{session_id}", dependencies=[Depends(verify_token)])
async def set_active_session(session_id: str):
    data = load_sessions()
    ids = [s["id"] for s in data["sessions"]]
    if session_id not in ids:
        raise HTTPException(status_code=404, detail="Session not found")
    data["active_session_id"] = session_id
    save_sessions(data)
    return {"active_session_id": session_id}


@app.delete("/rag/sessions/{session_id}", dependencies=[Depends(verify_token)])
async def delete_session(session_id: str):
    data = load_sessions()
    before = len(data["sessions"])
    data["sessions"] = [s for s in data["sessions"] if s["id"] != session_id]
    if len(data["sessions"]) == before:
        raise HTTPException(status_code=404, detail="Session not found")
    if data.get("active_session_id") == session_id:
        data["active_session_id"] = data["sessions"][0]["id"] if data["sessions"] else None
    save_sessions(data)
    return {"deleted": session_id}


@app.post("/rag/answer", response_model=AnswerResponse, dependencies=[Depends(verify_token)])
async def answer_endpoint(request: QuestionRequest, domain: str = "accounting"):
    domain_obj = get_domain(domain)
    if not state.initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")

    start_time = time.time()
    query_id = str(uuid.uuid4())
    logger.info("Question: %s", request.question)

    try:
        chat_history = build_chat_history(request.history)
        executor = agent.build_agent(domain_obj)
        response = executor.invoke({"input": request.question, "chat_history": chat_history})
        output_data = response.get("output", "")

        try:
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
            chunks_retrieved=len(citations),
            retrieved_texts=[str(c) for c in citations],
            rerank_scores=[],
            llm_latency_ms=total_latency * 0.9,
            total_latency_ms=total_latency,
            fallback_triggered=False,
        )
        return {
            "answer": answer,
            "latency_ms": round(total_latency, 2),
            "context_length": len(citations),
            "model": "gemini-flash-agent",
            "citations": citations,
        }
    except Exception as e:
        logger.error("Error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/stream", dependencies=[Depends(verify_token)])
async def stream_endpoint(request: QuestionRequest, domain: str = "accounting"):
    domain_obj = get_domain(domain)
    config = domain_obj.get_config()
    if not state.initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")

    async def event_generator() -> AsyncIterator[str]:
        start_time = time.time()
        try:
            chat_history = build_chat_history(request.history)
            search_query = request.question
            if chat_history:
                rephrase_res = state.rephrase_chain.invoke({"chat_history": chat_history, "question": request.question})
                search_query = core.clean_ai_content(rephrase_res.content)

            initial_docs = state.retriever.invoke(search_query)
            docs = state.reranker.rerank(search_query, initial_docs, top_k=int(config.get("top_k", 5)))
            context = "\n\n".join([doc.page_content for doc in docs])
            sources = list(set(doc.metadata.get("source", "") for doc in docs if doc.metadata.get("source")))
            if sources:
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            full_answer = ""
            async for chunk in state.qa_chain.astream({"context": context, "chat_history": chat_history, "question": request.question}):
                token = core.clean_ai_content(chunk.content)
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                    await asyncio.sleep(0)

            latency = (time.time() - start_time) * 1000
            yield f"data: {json.dumps({'type': 'metadata', 'latency_ms': round(latency, 2), 'context_length': len(context), 'model': 'gemini-flash-latest'})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("[STREAM] Error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

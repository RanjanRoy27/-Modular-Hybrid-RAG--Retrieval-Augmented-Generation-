import time
import logging
from typing import List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import core
import ingest
from langchain_core.messages import HumanMessage, AIMessage

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("rag_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RAG-API")

app = FastAPI(title="RAG MVP API", description="FastAPI wrapper for the Google Generative AI RAG system.")

class RAGState:
    def __init__(self):
        self.embeddings = None
        self.llm = None
        self.retriever = None
        self.rephrase_chain = None
        self.qa_chain = None
        self.initialized = False

    def initialize(self):
        if not core.validate_env():
            logger.error("API start failed: GOOGLE_API_KEY missing.")
            return

        logger.info("Initializing RAG components...")
        self.embeddings = core.get_embeddings()
        self.llm = core.get_llm()
        self.retriever, msg = core.load_retriever(self.embeddings)
        
        if self.retriever:
            self.rephrase_chain = core.rephrase_prompt_template() | self.llm
            self.qa_chain = core.qa_prompt_template() | self.llm
            self.initialized = True
            logger.info(f"RAG components ready. {msg}")
        else:
            logger.error(f"Failed to load retriever: {msg}")

state = RAGState()

@app.on_event("startup")
async def startup_event():
    state.initialize()

class QuestionRequest(BaseModel):
    question: str
    history: List[dict] = []

class AnswerResponse(BaseModel):
    answer: str
    latency_ms: float
    context_length: int
    model: str

class IngestResponse(BaseModel):
    status: str
    files_loaded: int
    chunks_created: int
    message: str

@app.post("/rag/ingest", response_model=IngestResponse)
async def ingest_endpoint():
    logger.info("Ingestion requested via API.")
    res = ingest.ingest_docs()
    
    if res["success"]:
        state.initialize() # Refresh with new vector index
        return {
            "status": "success",
            "files_loaded": res["files_loaded"],
            "chunks_created": res["chunks_created"],
            "message": "Ingestion complete and retriever re-initialized."
        }
    else:
        raise HTTPException(status_code=500, detail=res["error"])

@app.post("/rag/answer", response_model=AnswerResponse)
async def answer_endpoint(request: QuestionRequest):
    if not state.initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized or vector store missing.")

    start_time = time.time()
    logger.info(f"Question: {request.question}")

    try:
        chat_history = []
        for msg in request.history:
            if msg["role"] == "human":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))

        search_query = request.question
        if chat_history:
            rephrase_res = state.rephrase_chain.invoke({
                "chat_history": chat_history,
                "question": request.question
            })
            search_query = core.clean_ai_content(rephrase_res.content)
            logger.info(f"Rephrased Search Query: {search_query}")

        docs = state.retriever.invoke(search_query)
        context = "\n\n".join([doc.page_content for doc in docs])
        context_len = len(context)

        response = state.qa_chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "question": request.question
        })
        
        answer = core.clean_ai_content(response.content)
        latency = (time.time() - start_time) * 1000
        logger.info(f"Answer generated in {latency:.2f}ms")

        return {
            "answer": answer,
            "latency_ms": round(latency, 2),
            "context_length": context_len,
            "model": "gemini-flash-latest"
        }

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

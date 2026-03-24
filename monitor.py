import json
import logging
from datetime import datetime

logger = logging.getLogger("RAG-Monitor")
logger.setLevel(logging.INFO)

# Don't duplicate if already added
if not logger.handlers:
    file_handler = logging.FileHandler("rag_trace.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

def log_query(query_id: str, question: str, 
              chunks_retrieved: int, retrieved_texts: list, 
              rerank_scores: list, 
              llm_latency_ms: float, total_latency_ms: float, 
              fallback_triggered: bool):
    """
    Log structured JSON for a query execution trace.
    """
    trace = {
        "timestamp": datetime.now().isoformat(),
        "query_id": query_id,
        "question_length": len(question),
        "chunks_retrieved": chunks_retrieved,
        "rerank_scores": rerank_scores,
        "llm_latency_ms": round(llm_latency_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "fallback_triggered": fallback_triggered,
        # Store a snippet of the retrieved text for observability
        "chunk_summaries": [text[:100].replace("\n", " ") + "..." for text in retrieved_texts]
    }
    
    logger.info(json.dumps(trace))
    return trace

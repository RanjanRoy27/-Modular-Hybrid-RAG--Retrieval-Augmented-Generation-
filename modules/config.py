"""
config.py — Single Source of Truth for All Configuration
Loads .env, validates required keys, exposes typed constants.
Never import individual env vars directly elsewhere — always use this module.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
QDRANT_PATH: str = str(BASE_DIR / os.getenv("QDRANT_PATH", "qdrant_store"))
LOG_FILE: Path = BASE_DIR / "rag_trace.log"
SESSIONS_FILE: Path = BASE_DIR / "sessions.json"

# ── Auth ───────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
RAG_API_KEY: str = os.getenv("RAG_API_KEY", "")

# ── Models ─────────────────────────────────────────────────────────────────────
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Qdrant ─────────────────────────────────────────────────────────────────────
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")
VECTOR_SIZE: int = 768

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "20"))
TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", "5"))

# ── HyDE ───────────────────────────────────────────────────────────────────────
HYDE_THRESHOLD_WORDS: int = int(os.getenv("HYDE_THRESHOLD_WORDS", "8"))

# ── Hallucination Guard ────────────────────────────────────────────────────────
GROUNDING_THRESHOLD: float = float(os.getenv("GROUNDING_THRESHOLD", "0.6"))
HALLUCINATION_GUARD_DOMAINS: list = [
    d.strip()
    for d in os.getenv("HALLUCINATION_GUARD_DOMAINS", "healthcare,real_estate").split(",")
]

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── API ────────────────────────────────────────────────────────────────────────
MAX_QUESTION_LENGTH: int = int(os.getenv("MAX_QUESTION_LENGTH", "2000"))
MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "10"))
API_VERSION: str = "3.0.0"


def validate() -> bool:
    """Validate critical environment variables are set. Returns False if any are missing."""
    errors = []
    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY is not set in .env")
    if not RAG_API_KEY:
        errors.append("RAG_API_KEY is not set in .env")
    if errors:
        for e in errors:
            print(f"[CONFIG ERROR] {e}")
        return False
    return True


if __name__ == "__main__":
    print(f"BASE_DIR         : {BASE_DIR}")
    print(f"DATA_DIR         : {DATA_DIR}")
    print(f"QDRANT_PATH      : {QDRANT_PATH}")
    print(f"LLM_MODEL        : {LLM_MODEL}")
    print(f"EMBED_MODEL      : {EMBED_MODEL}")
    print(f"RERANKER_MODEL   : {RERANKER_MODEL}")
    print(f"TOP_K_RETRIEVAL  : {TOP_K_RETRIEVAL}")
    print(f"TOP_K_RERANK     : {TOP_K_RERANK}")
    print(f"HYDE_THRESHOLD   : {HYDE_THRESHOLD_WORDS} words")
    print(f"GROUNDING_THRESH : {GROUNDING_THRESHOLD}")
    print(f"GUARD_DOMAINS    : {HALLUCINATION_GUARD_DOMAINS}")
    print(f"Validation       : {'PASS' if validate() else 'FAIL'}")

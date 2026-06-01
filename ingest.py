import sys
from pathlib import Path

PLATFORM_ROOT = Path(__file__).resolve().parent / "rag-platform"
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

from ingest.ingest import ingest_docs, load_documents

__all__ = ["ingest_docs", "load_documents"]

if __name__ == "__main__":
    ingest_docs()

import hashlib
from datetime import datetime

from langchain_experimental.text_splitter import SemanticChunker


def split_documents(documents, embeddings, chunk_size: int | None = None, chunk_overlap: int | None = None):
    """Split documents using the existing semantic chunking strategy."""
    text_splitter = SemanticChunker(embeddings)
    texts = text_splitter.split_documents(documents)

    ingestion_timestamp = datetime.now().isoformat()
    for i, chunk in enumerate(texts):
        chunk.metadata["source_file"] = chunk.metadata.get("source", "unknown")
        chunk.metadata["page_number"] = chunk.metadata.get("page", 0)
        chunk.metadata["chunk_index"] = i
        chunk.metadata["ingestion_timestamp"] = ingestion_timestamp

        first_line = chunk.page_content.strip().split("\n")[0][:50]
        chunk.metadata["section_heading"] = first_line if first_line else "Unknown"

        chunk_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
        chunk.metadata["content_hash"] = chunk_hash

    return texts

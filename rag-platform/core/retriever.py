import os

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

q_client = None


def get_qdrant_client():
    global q_client
    if q_client is None:
        q_client = QdrantClient(path=os.getenv("QDRANT_PATH", "qdrant_store"))
    return q_client


def load_retriever(
    embeddings,
    top_k: int = 20,
    bm25_weight: float | None = None,
    semantic_weight: float | None = None,
):
    """Loads the Qdrant retriever. BM25 weights are accepted for domain routing compatibility."""
    try:
        client = get_qdrant_client()
        if not client.collection_exists("documents"):
            client.create_collection(
                collection_name="documents",
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name="documents",
            embedding=embeddings,
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        return retriever, "Using QDRANT search."

    except Exception as e:
        return None, f"Error loading vector store: {e}"

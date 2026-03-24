import logging
from typing import List
from sentence_transformers import CrossEncoder

logger = logging.getLogger("RAG-API")

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List, top_k: int = 5) -> List:
        """
        Rerank a list of LangChain Document objects based on the query.
        """
        if not documents:
            return []
            
        # Prepare inputs: list of (query, doc_text) pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Inject scores into metadata and sort
        for i, doc in enumerate(documents):
            doc.metadata["rerank_score"] = float(scores[i])
            
        # Sort documents by score descending
        reranked_docs = sorted(documents, key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)
        
        return reranked_docs[:top_k]

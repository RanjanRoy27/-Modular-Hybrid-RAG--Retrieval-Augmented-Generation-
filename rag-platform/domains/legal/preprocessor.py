from pathlib import Path
from typing import Any, Dict

from domains.base_domain import BaseDomain


def _load_config() -> Dict[str, Any]:
    path = Path(__file__).with_name("config.yaml")
    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return {
            "domain": "legal",
            "chunk_size": 800,
            "chunk_overlap": 100,
            "bm25_weight": 0.4,
            "semantic_weight": 0.6,
            "top_k": 5,
            "rerank": True,
            "system_prompt": "You are an expert legal assistant. Answer only based on the provided legal documents. Be precise about obligations, clauses, dates, and named parties. If unsure, say \"I cannot confirm this from the documents.\"",
            "document_types": ["pdf", "docx", "txt"],
        }


class LegalDomain(BaseDomain):
    def get_config(self) -> Dict[str, Any]:
        return _load_config()

    def preprocess_document(self, raw_text: str) -> str:
        return raw_text.strip()

    def get_system_prompt(self) -> str:
        return self.get_config()["system_prompt"]

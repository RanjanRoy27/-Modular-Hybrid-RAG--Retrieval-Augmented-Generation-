from abc import ABC, abstractmethod
from typing import List, Dict, Any

# STATUS: Canonical live API domain model. V3 modules/domain uses a separate experimental taxonomy.
class BaseDomain(ABC):
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return domain-specific config: prompts, chunk_size, model params"""
        pass

    @abstractmethod
    def preprocess_document(self, raw_text: str) -> str:
        """Domain-specific document cleaning before chunking"""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """System prompt injected into LLM for this domain"""
        pass

    def get_domain_name(self) -> str:
        return self.__class__.__module__.split(".")[-2]

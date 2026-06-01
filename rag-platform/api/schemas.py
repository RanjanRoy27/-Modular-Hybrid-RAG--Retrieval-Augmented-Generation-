from typing import List, Optional

from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str
    history: List[dict] = []
    session_id: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str
    latency_ms: float
    context_length: int
    model: str
    citations: List[dict] = []


class IngestResponse(BaseModel):
    status: str
    files_loaded: int
    chunks_created: int
    message: str
    session_id: str


class SessionCreate(BaseModel):
    name: Optional[str] = None


class MessageAppend(BaseModel):
    messages: List[dict]

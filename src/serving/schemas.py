"""Pydantic schemas for MedLLMOps API"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class AskRequest(BaseModel):
    """Request body for /ask endpoint"""
    model_config = ConfigDict(protected_namespaces=())
    
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Medical question",
        examples=["What are the side effects of metformin?"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of papers to retrieve",
    )


class Source(BaseModel):
    """Retrieved source paper"""
    rank: int
    title: str
    year: str
    journal: str
    authors: str
    pmid: str
    url: str
    similarity: float


class AskResponse(BaseModel):
    """Response from /ask endpoint"""
    model_config = ConfigDict(protected_namespaces=())
    
    question: str
    answer: str
    sources: List[Source]
    blocked: bool = False
    block_reason: Optional[str] = None
    is_emergency: bool = False
    is_personal_advice: bool = False
    pii_redacted: List[dict] = []
    n_sources: int = 0
    timestamp: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    rag_loaded: bool
    guardrails_loaded: bool
    vector_db_docs: int
    model_name: str
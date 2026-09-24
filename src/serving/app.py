"""FastAPI serving for MedLLMOps — Medical RAG API"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import os

from src.serving.schemas import (
    AskRequest, AskResponse, Source, HealthResponse,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load RAG pipeline at startup"""
    global pipeline
    
    try:
        logger.info("🚀 Loading MedLLMOps pipeline...")
        
        from src.rag.pipeline import MedicalRAGPipeline
        pipeline = MedicalRAGPipeline()
        
        logger.info("✅ MedLLMOps ready!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        pipeline = None
    
    yield
    
    logger.info("🛑 Shutting down")


app = FastAPI(
    title="MedLLMOps API",
    description=(
        "Production Medical RAG API with 4-layer safety guardrails. "
        "5,455 PubMed papers, 6,235 searchable chunks, cited answers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "MedLLMOps API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "description": "Medical RAG with safety guardrails",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """Health check"""
    rag_ok = pipeline is not None
    guardrails_ok = pipeline is not None and pipeline.guardrails is not None
    
    docs = 0
    if pipeline is not None and pipeline.retriever is not None:
        try:
            docs = pipeline.retriever.store.collection.count()
        except Exception:
            docs = 0
    
    return HealthResponse(
        status="healthy" if rag_ok else "unhealthy",
        rag_loaded=rag_ok,
        guardrails_loaded=guardrails_ok,
        vector_db_docs=docs,
        model_name=os.getenv("DEFAULT_MODEL", "openai/gpt-oss-20b"),
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(request: AskRequest):
    """
    Ask a medical question. Returns cited answer.
    
    Safety:
    - Prompt injection blocked
    - PII redacted
    - Medical disclaimer added
    - Emergency warning for dangerous queries
    """
    if pipeline is None:
        raise HTTPException(503, "Pipeline not loaded")
    
    try:
        result = pipeline.ask(request.question, top_k=request.top_k)
        
        sources = [
            Source(
                rank=s["rank"],
                title=s["title"][:200],
                year=s["year"],
                journal=s["journal"][:100],
                authors=s["authors"],
                pmid=s["pmid"],
                url=s["url"],
                similarity=round(s["similarity"], 4),
            )
            for s in result["sources"]
        ]
        
        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            sources=sources,
            blocked=result["blocked"],
            block_reason=result["block_reason"],
            is_emergency=result["is_emergency"],
            is_personal_advice=result["is_personal_advice"],
            pii_redacted=result["pii_redacted"],
            n_sources=result["n_sources"],
            timestamp=datetime.now().isoformat(),
        )
    
    except Exception as e:
        logger.error(f"Ask error: {e}")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.app:app", host="0.0.0.0", port=8000, reload=True)
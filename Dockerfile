# Base
FROM python:3.11-slim

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Step 1: Install CPU-only PyTorch FIRST (much smaller — 200 MB vs 800 MB+)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1

# Step 2: Install remaining deps
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Step 3: Download small spaCy model
RUN python -m spacy download en_core_web_sm

# Step 4: Copy source
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/processed/chunks.json ./data/processed/chunks.json
COPY data/processed/embeddings.npy ./data/processed/embeddings.npy
COPY data/processed/embeddings_meta.json ./data/processed/embeddings_meta.json

# Step 5: Build ChromaDB at image build time
RUN python -c "from src.rag.vector_store import VectorStore; \
    import json, numpy as np; \
    chunks = json.load(open('data/processed/chunks.json')); \
    emb = np.load('data/processed/embeddings.npy'); \
    s = VectorStore(); s.create_collection(reset=True); \
    s.add_chunks(chunks, emb); \
    print(f'✅ ChromaDB built: {s.collection.count()} docs')"

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
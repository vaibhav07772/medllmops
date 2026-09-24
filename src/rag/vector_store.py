"""ChromaDB vector store for medical RAG"""
import json
import numpy as np
import time
from pathlib import Path
from typing import List, Dict
from rich.console import Console
import chromadb
from chromadb.config import Settings

console = Console()

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "pubmed_abstracts"


class VectorStore:
    """ChromaDB wrapper for medical paper retrieval"""
    
    def __init__(self, persist_dir: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[cyan]🔧 Initializing ChromaDB...[/cyan]")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self.collection = None
    
    def create_collection(self, reset: bool = False):
        """Create or get collection"""
        if reset:
            try:
                self.client.delete_collection(self.collection_name)
                console.print(f"   🗑️  Deleted existing collection")
            except Exception:
                pass
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        console.print(f"   ✅ Collection: {self.collection_name}")
        console.print(f"   Documents: {self.collection.count()}")
    
    def add_chunks(self, chunks: List[Dict], embeddings: np.ndarray,
                    batch_size: int = 500):
        """Add chunks with embeddings to ChromaDB"""
        n = len(chunks)
        console.print(f"\n[cyan]📥 Adding {n} chunks to ChromaDB...[/cyan]")
        
        t0 = time.time()
        
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            batch_chunks = chunks[i:end]
            batch_embeddings = embeddings[i:end].tolist()
            
            ids = [c["chunk_id"] for c in batch_chunks]
            documents = [c["text"] for c in batch_chunks]
            metadatas = [{
                "pmid": c["pmid"],
                "title": c["title"][:500],
                "year": c.get("year", ""),
                "journal": c.get("journal", "")[:200],
                "topic": c.get("topic", ""),
                "url": c.get("url", ""),
                "authors": ", ".join(c.get("authors", [])[:3]),
            } for c in batch_chunks]
            
            self.collection.add(
                ids=ids,
                embeddings=batch_embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            
            console.print(f"   Batch {i//batch_size + 1}: added {end-i} chunks (total: {end})")
        
        elapsed = time.time() - t0
        console.print(f"\n[green]✅ Added {n} chunks in {elapsed:.1f}s[/green]")
        console.print(f"   Collection count: {self.collection.count()}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5,
                filter_dict: Dict = None) -> Dict:
        """Search similar chunks"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_dict,
        )
        return results


def main():
    console.print("\n[bold magenta]=" * 30)
    console.print("[bold magenta]🗄️  PHASE 2.3 — VECTOR STORE")
    console.print("[bold magenta]=" * 30 + "\n")
    
    # Load chunks + embeddings
    console.print("📂 Loading data...")
    
    with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load("data/processed/embeddings.npy")
    
    console.print(f"   Chunks: {len(chunks)}")
    console.print(f"   Embeddings: {embeddings.shape}")
    
    assert len(chunks) == len(embeddings), "Mismatch!"
    
    # Init store
    store = VectorStore()
    store.create_collection(reset=True)
    
    # Add chunks
    store.add_chunks(chunks, embeddings)
    
    # Test search
    console.print("\n[cyan]🔍 Testing search...[/cyan]")
    from src.rag.embeddings import EmbeddingGenerator
    gen = EmbeddingGenerator()
    
    test_query = "What are the latest treatments for type 2 diabetes?"
    q_emb = gen.encode([test_query], show_progress=False)[0]
    
    results = store.search(q_emb, top_k=3)
    
    console.print(f"\n[bold]Query:[/bold] {test_query}\n")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ), 1):
        console.print(f"[bold cyan]Result {i}[/bold cyan] (distance: {dist:.4f})")
        console.print(f"   Title: {meta['title'][:80]}...")
        console.print(f"   Year: {meta['year']}")
        console.print(f"   Text: {doc[:200]}...")
        console.print()


if __name__ == "__main__":
    main()
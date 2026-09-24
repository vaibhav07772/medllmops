"""Retriever — search ChromaDB for relevant chunks"""
import numpy as np
from typing import List, Dict
from rich.console import Console

from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStore

console = Console()


class Retriever:
    """Semantic retriever for medical papers"""
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.gen = EmbeddingGenerator()
        self.store = VectorStore()
        self.store.create_collection(reset=False)
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant chunks for a query"""
        k = top_k or self.top_k
        
        # Embed query
        q_emb = self.gen.encode([query], show_progress=False)[0]
        
        # Search
        results = self.store.search(q_emb, top_k=k)
        
        # Format results
        retrieved = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            retrieved.append({
                "rank": i + 1,
                "text": doc,
                "pmid": meta["pmid"],
                "title": meta["title"],
                "year": meta["year"],
                "journal": meta["journal"],
                "authors": meta["authors"],
                "topic": meta["topic"],
                "url": meta["url"],
                "similarity": 1 - dist,  # cosine similarity
                "distance": dist,
            })
        
        return retrieved


if __name__ == "__main__":
    console.print("\n[bold]🔍 Testing Retriever[/bold]\n")
    
    retriever = Retriever(top_k=5)
    
    query = "What are the side effects of metformin?"
    console.print(f"[cyan]Query:[/cyan] {query}\n")
    
    results = retriever.retrieve(query)
    
    for r in results:
        console.print(f"[bold]Rank {r['rank']}[/bold] (sim: {r['similarity']:.4f})")
        console.print(f"   Title: {r['title'][:80]}")
        console.print(f"   Year: {r['year']} | Journal: {r['journal'][:40]}")
        console.print(f"   PMID: {r['pmid']}")
        console.print(f"   Text: {r['text'][:200]}...\n")
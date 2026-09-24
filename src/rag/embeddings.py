"""Generate embeddings using sentence-transformers (local, free)"""
import json
import numpy as np
import time
from pathlib import Path
from typing import List
from rich.console import Console
from sentence_transformers import SentenceTransformer

console = Console()


class EmbeddingGenerator:
    """Generate embeddings for text chunks"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        console.print(f"[cyan]🔧 Loading model: {model_name}[/cyan]")
        t0 = time.time()
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()
        console.print(f"   ✅ Loaded in {time.time()-t0:.1f}s | Dimension: {self.dim}")
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True):
        """Generate embeddings for list of texts"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # for cosine similarity
        )


def main():
    console.print("\n[bold magenta]=" * 30)
    console.print("[bold magenta]🧠 PHASE 2.2 — EMBEDDINGS")
    console.print("[bold magenta]=" * 30 + "\n")
    
    # Load chunks
    input_file = Path("data/processed/chunks.json")
    console.print(f"📂 Loading {input_file}...")
    
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    console.print(f"   Loaded {len(chunks)} chunks")
    
    # Generate embeddings
    console.print("\n[cyan]🔧 Generating embeddings...[/cyan]")
    gen = EmbeddingGenerator()
    
    texts = [c["text"] for c in chunks]
    t0 = time.time()
    embeddings = gen.encode(texts, batch_size=32, show_progress=True)
    elapsed = time.time() - t0
    
    console.print(f"\n[green]✅ Generated {len(embeddings)} embeddings in {elapsed/60:.1f} min[/green]")
    console.print(f"   Shape: {embeddings.shape}")
    console.print(f"   Size: {embeddings.nbytes / 1024**2:.1f} MB")
    
    # Save
    output_dir = Path("data/processed")
    output_file = output_dir / "embeddings.npy"
    np.save(output_file, embeddings)
    console.print(f"\n💾 Saved: {output_file}")
    
    # Save metadata for reference
    meta = {
        "model_name": gen.model_name,
        "dimension": gen.dim,
        "n_chunks": len(chunks),
        "shape": list(embeddings.shape),
    }
    with open(output_dir / "embeddings_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    console.print(f"💾 Saved: {output_dir / 'embeddings_meta.json'}")


if __name__ == "__main__":
    main()
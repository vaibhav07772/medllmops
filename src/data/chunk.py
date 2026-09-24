"""Chunk PubMed papers into smaller pieces for RAG"""
import json
import re
from pathlib import Path
from typing import List, Dict
from rich.console import Console

console = Console()


def clean_text(text: str) -> str:
    """Clean text — remove extra whitespace"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text.strip()


def chunk_by_sentences(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Chunk text by sentences with character-based size.
    Approximates token count (1 token ≈ 4 chars).
    """
    if not text or len(text) < 50:
        return []
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current = ""
    max_chars = chunk_size * 4  # ~512 tokens ≈ 2048 chars
    overlap_chars = overlap * 4
    
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_chars and current:
            chunks.append(current.strip())
            # Overlap: keep last portion
            current = current[-overlap_chars:] + " " + sent if overlap_chars > 0 else sent
        else:
            current = current + " " + sent if current else sent
    
    if current.strip():
        chunks.append(current.strip())
    
    return chunks


def chunk_papers(papers: List[Dict], chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
    """
    Chunk all papers. Each chunk has metadata.
    """
    all_chunks = []
    
    for i, paper in enumerate(papers):
        if (i + 1) % 500 == 0:
            console.print(f"   Processed {i+1}/{len(papers)} papers...")
        
        # Combine title + abstract for context
        title = clean_text(paper.get("title", ""))
        abstract = clean_text(paper.get("abstract", ""))
        
        if not abstract:
            continue
        
        # Full text with title
        full_text = f"{title}. {abstract}"
        
        chunks = chunk_by_sentences(full_text, chunk_size, overlap)
        
        for j, chunk_text in enumerate(chunks):
            chunk = {
                "chunk_id": f"{paper['pmid']}_{j}",
                "pmid": paper["pmid"],
                "text": chunk_text,
                "title": title,
                "year": paper.get("year", ""),
                "journal": paper.get("journal", ""),
                "authors": paper.get("authors", []),
                "topic": paper.get("topic", ""),
                "url": paper.get("url", ""),
                "chunk_index": j,
                "total_chunks": len(chunks),
            }
            all_chunks.append(chunk)
    
    return all_chunks


def main():
    console.print("\n[bold magenta]=" * 30)
    console.print("[bold magenta]📝 PHASE 2.1 — CHUNKING")
    console.print("[bold magenta]=" * 30 + "\n")
    
    # Load papers
    input_file = Path("data/raw/pubmed_papers.json")
    console.print(f"📂 Loading {input_file}...")
    
    with open(input_file, "r", encoding="utf-8") as f:
        papers = json.load(f)
    
    console.print(f"   Loaded {len(papers)} papers")
    
    # Chunk
    console.print("\n[cyan]🔪 Chunking (512 tokens, 50 overlap)...[/cyan]")
    chunks = chunk_papers(papers, chunk_size=512, overlap=50)
    
    console.print(f"\n[green]✅ Total chunks: {len(chunks)}[/green]")
    console.print(f"   Avg chunks/paper: {len(chunks)/len(papers):.1f}")
    
    # Save
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "chunks.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    size_mb = output_file.stat().st_size / 1024**2
    console.print(f"\n💾 Saved: {output_file}")
    console.print(f"   Size: {size_mb:.1f} MB")
    
    # Stats
    chunk_lengths = [len(c["text"]) for c in chunks]
    console.print(f"\n📊 Chunk stats:")
    console.print(f"   Min length: {min(chunk_lengths)} chars")
    console.print(f"   Max length: {max(chunk_lengths)} chars")
    console.print(f"   Avg length: {sum(chunk_lengths)/len(chunk_lengths):.0f} chars")


if __name__ == "__main__":
    main()
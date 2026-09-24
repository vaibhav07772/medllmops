"""
Collect PubMed papers across medical topics.
Saves to data/raw/pubmed_papers.json
"""
import json
import time
from pathlib import Path
from rich.console import Console

from src.data.pubmed_api import PubMedClient

console = Console()

# Medical topics to collect
TOPICS = [
    ("diabetes mellitus type 2 treatment", 800),
    ("cardiovascular disease prevention", 800),
    ("cancer immunotherapy PD-L1", 800),
    ("COVID-19 vaccine efficacy", 800),
    ("depression anxiety treatment", 700),
    ("alzheimer disease biomarkers", 700),
    ("hypertension management guidelines", 700),
    ("antibiotic resistance mechanisms", 700),
]

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "pubmed_papers.json"


def main():
    console.print("\n[bold magenta]=" * 30)
    console.print("[bold magenta]🏥 PHASE 1 — PUBMED DATA COLLECTION")
    console.print("[bold magenta]=" * 30 + "\n")
    
    client = PubMedClient()
    all_papers = []
    seen_pmids = set()
    
    t0 = time.time()
    
    for i, (query, max_results) in enumerate(TOPICS, 1):
        console.print(f"\n[bold yellow]📚 Topic {i}/{len(TOPICS)}[/bold yellow]")
        
        try:
            papers = client.search_and_fetch(query, max_results=max_results)
        except Exception as e:
            console.print(f"[red]❌ Failed: {e}[/red]")
            continue
        
        # Deduplicate by PMID
        new_papers = 0
        for p in papers:
            if p["pmid"] and p["pmid"] not in seen_pmids:
                seen_pmids.add(p["pmid"])
                p["topic"] = query
                all_papers.append(p)
                new_papers += 1
        
        console.print(f"[green]   → {new_papers} new papers (total: {len(all_papers)})[/green]")
    
    elapsed = time.time() - t0
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)
    
    size_mb = OUTPUT_FILE.stat().st_size / 1024**2
    
    console.print(f"\n[bold green]=" * 30)
    console.print(f"[bold green]🎉 COLLECTION COMPLETE")
    console.print(f"[bold green]=" * 30)
    console.print(f"   Total papers: [bold]{len(all_papers)}[/bold]")
    console.print(f"   Unique PMIDs: {len(seen_pmids)}")
    console.print(f"   Saved: {OUTPUT_FILE}")
    console.print(f"   Size: {size_mb:.1f} MB")
    console.print(f"   Time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
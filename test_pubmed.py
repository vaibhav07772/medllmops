"""Quick test — fetch 20 papers"""
from src.data.pubmed_api import PubMedClient
import json

client = PubMedClient()
papers = client.search_and_fetch("diabetes treatment", max_results=20)

print(f"\n✅ Fetched: {len(papers)} papers\n")
for p in papers[:3]:
    print(f"📄 {p['title'][:100]}")
    print(f"   Year: {p['year']}, Journal: {p['journal'][:50]}")
    print(f"   Authors: {', '.join(p['authors'][:2])}")
    print(f"   Abstract: {p['abstract'][:150]}...")
    print()
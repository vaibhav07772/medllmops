"""
PubMed E-utilities API client.
Free API — no key needed. Rate limit: 3 requests/sec.
Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""
import requests
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from rich.console import Console

console = Console()

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedClient:
    """Client for NCBI PubMed E-utilities API"""
    
    def __init__(self, email: str = "vaibhav@example.com", tool: str = "medllmops"):
        self.email = email
        self.tool = tool
        self.session = requests.Session()
        self.session.params = {"email": email, "tool": tool}
    
    def _rate_limit(self):
        """Respect 3 requests/sec limit"""
        time.sleep(0.4)
    
    def search(self, query: str, max_results: int = 100) -> List[str]:
        """
        Search PubMed for PMIDs matching query.
        
        Returns list of PMIDs (strings).
        """
        self._rate_limit()
        
        url = f"{BASE_URL}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        
        data = resp.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        return pmids
    
    def fetch_details(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch full metadata for PMIDs (batch).
        Returns list of paper dicts.
        """
        if not pmids:
            return []
        
        self._rate_limit()
        
        url = f"{BASE_URL}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        
        resp = self.session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        
        return self._parse_xml(resp.text)
    
    def _parse_xml(self, xml_text: str) -> List[Dict]:
        """Parse PubMed XML response"""
        papers = []
        
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            console.print(f"[red]XML parse error: {e}[/red]")
            return papers
        
        for article in root.findall(".//PubmedArticle"):
            try:
                paper = self._extract_article(article)
                if paper:
                    papers.append(paper)
            except Exception as e:
                console.print(f"[yellow]Skip article: {e}[/yellow]")
                continue
        
        return papers
    
    def _extract_article(self, article) -> Optional[Dict]:
        """Extract fields from a single article"""
        # PMID
        pmid_elem = article.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else None
        
        # Title
        title_elem = article.find(".//ArticleTitle")
        title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""
        
        # Abstract
        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join("".join(p.itertext()).strip() for p in abstract_parts)
        
        # Authors
        authors = []
        for author in article.findall(".//Author")[:5]:
            last = author.find("LastName")
            first = author.find("ForeName")
            if last is not None:
                name = last.text or ""
                if first is not None and first.text:
                    name = f"{first.text} {name}"
                authors.append(name)
        
        # Journal
        journal_elem = article.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else ""
        
        # Year
        year_elem = article.find(".//PubDate/Year")
        if year_elem is None:
            year_elem = article.find(".//PubDate/MedlineDate")
        year = year_elem.text[:4] if year_elem is not None and year_elem.text else ""
        
        # MeSH terms (medical keywords)
        mesh_terms = []
        for mesh in article.findall(".//MeshHeading/DescriptorName")[:10]:
            if mesh.text:
                mesh_terms.append(mesh.text)
        
        if not abstract or not title:
            return None
        
        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "journal": journal,
            "year": year,
            "mesh_terms": mesh_terms,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        }
    
    def search_and_fetch(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Combined: search + fetch in batches of 100.
        """
        console.print(f"\n[cyan]🔍 Query:[/cyan] {query}")
        console.print(f"   Target: {max_results} papers")
        
        # Search for PMIDs
        pmids = self.search(query, max_results=max_results)
        console.print(f"   Found: {len(pmids)} PMIDs")
        
        # Fetch in batches of 100
        all_papers = []
        batch_size = 100
        
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            console.print(f"   Fetching batch {i//batch_size + 1}/{(len(pmids)-1)//batch_size + 1}...")
            papers = self.fetch_details(batch)
            all_papers.extend(papers)
        
        console.print(f"[green]✅ Total papers: {len(all_papers)}[/green]")
        return all_papers
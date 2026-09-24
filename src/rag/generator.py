"""Answer generator — LLM se answer with citations"""
import os
from typing import List, Dict
from rich.console import Console
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
console = Console()


RAG_PROMPT = """You are a medical AI assistant. Answer the user's question based ONLY on the provided context from medical research papers.

CRITICAL RULES:
1. If the context does not contain the answer, say "I don't have enough information in my knowledge base to answer this."
2. ALWAYS cite sources using [1], [2], [3] format when you use information.
3. Be precise and factual. Do NOT hallucinate.
4. Add a disclaimer that this is not medical advice.

CONTEXT:
{context}

QUESTION:
{question}

Provide a clear, well-structured answer with citations [1], [2], etc."""


class AnswerGenerator:
    """Generate answers with citations from retrieved context"""
    
    def __init__(self, model: str = "openai/gpt-oss-120b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY missing in .env")
        
        self.llm = ChatGroq(
            model=model,
            api_key=api_key,
            temperature=0.1,
            max_tokens=1000,
        )
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks with numbered citations"""
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[{i}] Title: {c['title']}\n"
                f"    Year: {c['year']} | Journal: {c['journal']}\n"
                f"    Text: {c['text']}\n"
            )
        return "\n".join(parts)
    
    def _format_sources(self, chunks: List[Dict]) -> str:
        """Format source list"""
        lines = ["\n\n---\n**📚 Sources:**\n"]
        seen = set()
        for i, c in enumerate(chunks, 1):
            if c["pmid"] in seen:
                continue
            seen.add(c["pmid"])
            lines.append(
                f"[{i}] {c['authors'][:80]} ({c['year']}). "
                f"*{c['title']}*. {c['journal']}. "
                f"[PubMed]({c['url']})"
            )
        return "\n".join(lines)
    
    def generate(self, question: str, retrieved_chunks: List[Dict]) -> Dict:
        """Generate answer with citations"""
        context = self._format_context(retrieved_chunks)
        
        answer = self.chain.invoke({
            "context": context,
            "question": question,
        })
        
        sources = self._format_sources(retrieved_chunks)
        full_answer = answer + sources
        
        return {
            "answer": full_answer,
            "n_sources": len(retrieved_chunks),
        }


if __name__ == "__main__":
    from src.rag.retriever import Retriever
    
    console.print("\n[bold]🧪 Testing Full RAG Pipeline[/bold]\n")
    
    question = "What are the side effects of metformin?"
    console.print(f"[cyan]Question:[/cyan] {question}\n")
    
    # Retrieve
    retriever = Retriever(top_k=3)
    chunks = retriever.retrieve(question)
    console.print(f"✅ Retrieved {len(chunks)} chunks\n")
    
    # Generate
    generator = AnswerGenerator()
    result = generator.generate(question, chunks)
    
    console.print("\n[bold green]="*30)
    console.print("[bold green]ANSWER:")
    console.print("[bold green]="*30 + "\n")
    console.print(result["answer"])
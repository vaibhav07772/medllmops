"""
RAGAS Evaluation for Medical RAG Pipeline.
Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
"""
import json
import os
import time
from pathlib import Path
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

load_dotenv()
console = Console()


def run_evaluation(
    n_questions: int = 10,
    output_file: str = "reports/ragas_eval_results.json",
):
    """
    Run RAGAS evaluation on the medical RAG pipeline.
    """
    console.print("\n[bold magenta]=" * 30)
    console.print("[bold magenta]📊 PHASE 4 — RAGAS EVALUATION")
    console.print("[bold magenta]=" * 30 + "\n")
    
    # ─── 1. Load benchmark ───
    console.print("[cyan]📂 Loading benchmark questions...[/cyan]")
    with open("data/benchmarks/eval_questions.json", "r") as f:
        benchmark = json.load(f)
    
    benchmark = benchmark[:n_questions]
    console.print(f"   Loaded {len(benchmark)} questions")
    
    # ─── 2. Initialize RAG pipeline ───
    console.print("\n[cyan]🔧 Initializing RAG pipeline...[/cyan]")
    from src.rag.retriever import Retriever
    from src.rag.generator import AnswerGenerator
    
    retriever = Retriever(top_k=3)
    generator = AnswerGenerator()
    
    # ─── 3. Collect answers + contexts ───
    console.print(f"\n[cyan]🧠 Running RAG on {len(benchmark)} questions...[/cyan]")
    t0 = time.time()
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for i, item in enumerate(benchmark, 1):
        q = item["question"]
        gt = item["ground_truth"]
        
        console.print(f"\n[dim]Q{i}/{len(benchmark)}: {q[:60]}...[/dim]")
        
        # Retrieve
        chunks = retriever.retrieve(q, top_k=3)
        
        # Generate
        try:
            gen_result = generator.generate(q, chunks)
            answer = gen_result["answer"].split("---")[0]
        except Exception as e:
            console.print(f"[red]   Error: {e}[/red]")
            continue
        
        questions.append(q)
        answers.append(answer)
        contexts.append([c["text"] for c in chunks])
        ground_truths.append(gt)
    
    elapsed = time.time() - t0
    console.print(f"\n[green]✅ Collected {len(questions)} RAG outputs in {elapsed/60:.1f} min[/green]")
    
    # ─── 4. Run RAGAS ───
    console.print("\n[cyan]📊 Running RAGAS evaluation...[/cyan]")
    console.print("[yellow]⚠️  This takes 5-10 min. Groq free tier rate limits may cause some timeouts.[/yellow]")
    
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    # Setup Groq LLM for RAGAS with longer timeout
    groq_llm = ChatGroq(
        model=os.getenv("DEFAULT_MODEL", "openai/gpt-oss-20b"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0,
        timeout=120,
        max_retries=5,
    )
    ragas_llm = LangchainLLMWrapper(groq_llm)
    
    # Setup embeddings
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    
    # Configure metrics
    for m in [faithfulness, answer_relevancy, context_precision, context_recall]:
        m.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings
    
    # Create dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            raise_exceptions=False,   # Don't crash on individual failures
        )
    except Exception as e:
        console.print(f"[red]❌ RAGAS failed: {e}[/red]")
        return None
    
    # ─── 5. Display results ───
    console.print("\n[bold green]=" * 30)
    console.print("[bold green]📊 RAGAS RESULTS")
    console.print("[bold green]=" * 30 + "\n")
    
    # ✅ RAGAS 0.2.x compatible
    try:
        result_df = result.to_pandas()
        metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        available_cols = [c for c in metric_cols if c in result_df.columns]
        
        scores = {}
        for col in available_cols:
            # Drop NaN (failed evaluations)
            valid_vals = result_df[col].dropna()
            if len(valid_vals) > 0:
                scores[col] = float(valid_vals.mean())
            else:
                scores[col] = 0.0
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not extract scores: {e}[/yellow]")
        console.print(f"[dim]Raw result: {result}[/dim]")
        return None
    
    # Display table
    table = Table(title="RAGAS Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="bold green")
    table.add_column("Target", style="yellow")
    table.add_column("Status", style="magenta")
    
    targets = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_precision": 0.70,
        "context_recall": 0.75,
    }
    
    for metric, score in scores.items():
        target = targets.get(metric, 0.70)
        status = "✅ PASS" if score >= target else "⚠️ BELOW"
        table.add_row(
            metric,
            f"{score:.4f}",
            f"{target:.2f}",
            status,
        )
    
    console.print(table)
    
    # ─── 6. Save results ───
    Path("reports").mkdir(exist_ok=True)
    
    output_data = {
        "scores": scores,
        "n_questions": len(questions),
        "elapsed_min": elapsed / 60,
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    console.print(f"\n💾 Saved: {output_file}")
    
    return output_data


if __name__ == "__main__":
    run_evaluation(n_questions=10)   # ✅ Reduced from 20 to 10
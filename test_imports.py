"""Phase 0 — Verify all imports work"""
print("🔍 Testing imports...\n")

# Core
import numpy, pandas, yaml
print("✅ Core: numpy, pandas, yaml")

# LLM
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
print("✅ LLM: langchain, groq")

# Embeddings
from sentence_transformers import SentenceTransformer
print("✅ Embeddings: sentence-transformers")

# Vector DB
import chromadb
print("✅ Vector DB: chromadb")

# RAG Eval
import ragas
print("✅ RAG Eval: ragas")

# Guardrails
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
print("✅ Guardrails: presidio")

# API
from fastapi import FastAPI
print("✅ API: fastapi")

# Utils
import rich, tqdm
print("✅ Utils: rich, tqdm")

print("\n🎉 All imports successful!")

# Test Groq connection
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if api_key and api_key.startswith("gsk_"):
    print(f"✅ GROQ_API_KEY found (length: {len(api_key)})")
else:
    print("❌ GROQ_API_KEY missing or invalid")
# Local Retail RAG with Ollama and Chroma

A simple local Retrieval-Augmented Generation (RAG) project that lets you ask natural-language questions about a retail dataset.

## Overview

This project:
- loads retail transaction data from a CSV file
- converts rows into text documents
- creates embeddings locally using Ollama
- stores vectors in Chroma
- retrieves relevant records for a user query
- uses a local LLM to generate a grounded answer

## Tech Stack

- Python
- Ollama
- Llama 3.2 3B
- embeddinggemma
- ChromaDB
- pandas
- requests

## Project Structure

```text
ai-retail-rag/
├─ data/
│  └─ online_retail.csv
├─ chroma_db/
├─ .venv/
├─ app.py
├─ ingest.py
├─ query.py
├─ requirements.txt
└─ README.md
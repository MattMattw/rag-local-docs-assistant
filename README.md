# Local RAG Assistant with Ollama

Proof of Concept of a Retrieval-Augmented Generation (RAG) system using
a local LLM via Ollama.

## Stack
- Python
- Ollama (Llama 3.1)
- LangChain
- FAISS

## Features
- Load local PDF documents
- Vector-based retrieval
- Context-aware LLM responses

## How to run
```bash
pip install -r requirements.txt
ollama run llama3.1
python app.py

# DocSage

**An evaluation-driven, multimodal RAG agent for AWS technical documentation.**

Ask questions in plain English. Get answers grounded in the actual documentation, with source citations, diagram understanding, and automatic quality scoring.

---

## Live Demo

> **[Launch DocSage on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/docsage)**
> *(Replace with your actual link after deployment)*

---

## What makes this different from a basic "chat with PDF" demo

| Feature | Most RAG demos | DocSage |
|---|---|---|
| Retrieval method | Vector search only | BM25 + vector search + cross-encoder reranking |
| Input types | Text only | Text, tables, architecture diagrams |
| Diagram handling | Skip images | Llama 4 Scout vision captions each diagram |
| Agent layer | None | LangGraph ReAct agent with MCP tool-use |
| Hallucination control | None | Explicit refusal phrase when context is insufficient |
| Quality measurement | Vibes | RAGAS (faithfulness, relevancy, precision/recall) |
| Deployment | Local only | Public URL on HF Spaces (free) |

---

## Architecture

```
PDF files
    │
    ▼
[Ingestion]          extract.py → chunk.py → caption.py (Groq vision)
    │
    ▼
[Indexing]           embed.py (bge-small, CPU) → Chroma DB
                     sparse.py (BM25) → bm25_index.pkl
    │
    ▼
[Query]
  User question
    │
    ▼
[Agent: LangGraph]   ReAct loop: reason → call tool → observe → repeat
    │
    ├── Tool: search_documentation (FastMCP → hybrid search → rerank)
    └── Tool: estimate_monthly_cost (deterministic calculator)
    │
    ▼
[Generation]         Groq Llama 3.3 70B → cited answer or refusal
    │
    ▼
[Evaluation]         RAGAS scores + latency p50/p95
```

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/YOUR_USERNAME/docsage.git
cd docsage
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Add your Groq API key (free at console.groq.com)
cp .env.example .env
# Edit .env and set GROQ_API_KEY=...

# 3. Add AWS PDF docs to data/raw/
# (download free whitepapers from aws.amazon.com/whitepapers)

# 4. Build the index
python scripts/build_index.py

# 5. Run the app
streamlit run app.py
```

Full setup guide: [docs/SETUP_AND_RUN.md](docs/SETUP_AND_RUN.md)

---

## Tech Stack

Python · PyMuPDF · pdfplumber · sentence-transformers (BAAI/bge-small) · Chroma · rank-bm25 · Groq (Llama 3.3 70B + Llama 4 Scout) · LangGraph · FastMCP · RAGAS · FastAPI · Streamlit · Docker · Hugging Face Spaces

---

## Documentation

| File | Contents |
|---|---|
| [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | Project overview, rationale, future scope |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | Every term, abbreviation, and concept explained |
| [docs/SETUP_AND_RUN.md](docs/SETUP_AND_RUN.md) | Step-by-step local development guide |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | How to deploy to HF Spaces and get a resume link |

---

## Note on `data/processed/`

The Chroma DB and BM25 index are committed to this repo. This is intentional: the HF Spaces deployment serves the app directly without running ingestion, so the pre-built index must be present. This is the opposite of the usual "don't commit generated files" advice — it is a deliberate tradeoff for a public demo.

---

## Author

**Sunny Khandekar** · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)

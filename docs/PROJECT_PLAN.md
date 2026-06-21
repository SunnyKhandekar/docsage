# DocSage — Project Plan

## Project Name
**DocSage**

## Full Title
DocSage: An Evaluation-Driven, Multimodal RAG Agent for AWS Technical Documentation

---

## Description in Plain Terms

DocSage is an AI assistant that reads AWS technical documentation — PDFs, diagrams, and pricing tables — and lets you ask questions about it in plain English. It answers only using what's actually in those documents, tells you exactly which page it pulled the answer from, and says "I don't have enough information" instead of making something up.

Unlike a simple chatbot:
- It understands architecture **diagrams** (not just text paragraphs)
- It can use **tools** — like looking up a cost estimate — instead of guessing
- It **grades its own answers** using a 30-question test set, so you have hard numbers for quality

The domain is AWS docs because Sunny holds AWS certifications, which gives a clear story to recruiters. But the architecture is domain-agnostic — swap the PDFs, get a different expert assistant.

---

## Tech Stack

| Layer | Tool | Why this choice |
|---|---|---|
| PDF parsing (text + tables) | PyMuPDF, pdfplumber | Fast, rule-based, no heavy ML model needed |
| PDF parsing (images) | PyMuPDF (fitz) | Extracts embedded images for captioning |
| Image/diagram understanding | Llama 4 Scout via Groq (free) | Vision model, converts diagrams to searchable text |
| Text embeddings | BAAI/bge-small-en-v1.5 (sentence-transformers) | Runs on CPU, compact, high quality for size |
| Vector database | Chroma (local, persistent) | Free, runs locally, no Docker required |
| Keyword search | rank_bm25 (BM25Okapi) | Complements vector search for exact-term queries |
| Result reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Lightweight second pass; corrects vector search noise |
| Answer generation | Llama 3.3 70B via Groq (free) | Strong general model, free API, fast on Groq LPUs |
| Agent orchestration | LangGraph (create_react_agent) | Decides: retrieve, call tool, or answer directly |
| Tool protocol | FastMCP (official MCP Python SDK) | Exposes retriever + cost tool as MCP-standard tools |
| Evaluation | RAGAS | Scores faithfulness, relevancy, precision, recall |
| REST API | FastAPI + Uvicorn | Wraps the pipeline as a proper HTTP service |
| Frontend | Streamlit | Clean chat UI, shows source citations and scores |
| Deployment | Hugging Face Spaces (Docker SDK) | Free public URL — put it on your resume |

---

## How It Works — Step by Step

### Offline: Build the Index (run once)
```
PDF file
  → extract_pdf()        # PyMuPDF: text per page, images saved to disk
  → chunk_page()         # Split text into 500-word chunks, 100-word overlap
  → caption_image()      # Groq vision: describe each diagram in words
  → embed_texts()        # sentence-transformers: turn chunks into vectors
  → add_chunks()         # Store vectors + metadata in Chroma
  → build_bm25_index()   # Build keyword index, save to disk
```

### Online: Answer a Question
```
User question
  → Agent (LangGraph ReAct loop)
      → Tool: search_documentation(query)
          → hybrid_search()    # Dense (Chroma) + Sparse (BM25) combined via RRF
          → rerank()           # CrossEncoder picks the best 5 of top 20
          → return top chunks + source metadata
      → Tool: estimate_monthly_cost(rate, hours)   # if cost question
      → generate()             # Groq: answer strictly from context, cite sources
  → Response: answer + [source_doc, page X] citations
```

### Periodic: Evaluate
```
eval_set.json (30 Q&A pairs)
  → run each question through the pipeline
  → RAGAS: score faithfulness, answer_relevancy, context_precision, context_recall
  → log latency p50 / p95
  → print report
```

---

## Why This Project Is Strong for a Portfolio

1. **Multimodal** — handles diagrams and tables, not just plain paragraphs. Most student RAG projects ignore images entirely.
2. **Hybrid retrieval** — BM25 + dense vectors + reranking. Any interviewer who knows RAG will ask about this; you have a real answer.
3. **Evaluation harness** — RAGAS metrics mean you can say "my system scores 0.87 faithfulness on 30 test questions." That's a number, not a vibe.
4. **MCP tool-use** — the emerging standard for how AI agents interact with tools. Being able to explain MCP in an interview puts you ahead of most candidates in 2026.
5. **Guardrails** — the system explicitly refuses to answer when the context doesn't support a confident response. Hallucination control is a real production concern and you've addressed it.
6. **Zero cost** — Groq free tier, Chroma local, HF Spaces free. Engineering judgment includes not wasting money.
7. **Personal narrative** — AWS certs justify the domain. RLHF/eval work at Innodata directly maps to the evaluation harness. One project, two experience points connected.

---

## Future Scope

| Idea | What it adds |
|---|---|
| Voice interface | Reuse WebSpeech + Flask from your interview chatbot project |
| Live AWS Pricing API tool | Replace static cost table with real-time API call via MCP |
| Second domain (e.g., insurance policies) | Proves architecture is general, not domain-locked |
| Langfuse / MLflow tracing | Real observability: trace every retrieval and generation step |
| Fine-tune a small model on eval Q&As | Cut latency and API dependency; LoRA on Mistral 7B |
| Streaming responses | FastAPI + Streamlit SSE for typewriter-style output |
| Feedback loop | Let users rate answers; pipe ratings back into eval dataset |

---

## Building This on a 4GB RAM Laptop

The core principle: **never load a large model locally**. All generation and vision happen via Groq API.

| Risk | Mitigation |
|---|---|
| Embedding a large corpus eats RAM | Process one PDF at a time; call `gc.collect()` between docs; batch size = 8 chunks |
| unstructured.io hi_res mode loads detectron2 | Never use it. PyMuPDF + pdfplumber is the stack. |
| Docker Desktop competing for RAM | Skip it on your laptop. HF Spaces builds the container server-side. |
| sentence-transformers + torch overhead | Use ONNX backend with int8 quantization if RAM is tight |
| Chroma loading full index into memory | Chroma uses sqlite3 + memory-mapped files; under 500MB for typical AWS doc set |

Working pattern: close all browser tabs before running `build_index.py`. Keep VS Code's extension set minimal. If on Windows, ensure a swap file is configured.

---

## What You Need to Learn

### Concepts (in the order you'll encounter them)
1. How vector embeddings represent semantic meaning
2. Why chunking strategy (size + overlap) matters for retrieval quality
3. Why BM25 + vectors beats either alone (complementary failure modes)
4. What a cross-encoder reranker actually computes vs a bi-encoder
5. Reciprocal Rank Fusion (RRF) — how to merge two ranked lists fairly
6. The ReAct agent loop: Reason → Act → Observe → repeat
7. What MCP is, why it exists, and how stdio transport works
8. RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall
9. FastAPI: routes, Pydantic models, async handlers
10. Streamlit: session state, chat_message, spinner

### Tools (hands-on time needed)
- `sentence-transformers` — 1 day to understand, encode, and benchmark
- `chromadb` — half a day; the API is small
- `rank_bm25` — 1 hour; it's three lines of code
- `LangGraph` — 2-3 days; the graph abstraction takes time to feel natural
- `mcp` (Python SDK) — 1-2 days; follow the FastMCP quickstart
- `ragas` — 1 day; focus on what each metric actually measures
- `FastAPI` — 2 days if new to it; 4 hours if you've used Flask
- `Streamlit` — 1 day for the basics needed here
- Git + GitHub + HF Spaces CLI — 1 day if rusty

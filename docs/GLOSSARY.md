# DocSage — Glossary

Every abbreviation, term, model name, library, concept, and jargon used in this project. Organised by category. Read this once, then use it as a reference.

---

## AI / ML Concepts

| Term | Full Form (if any) | Plain Meaning |
|---|---|---|
| RAG | Retrieval-Augmented Generation | A technique where an LLM answers questions using specific documents you feed it, instead of relying on what it memorised during training. Makes answers grounded and citable. |
| LLM | Large Language Model | A neural network trained on massive text data that can generate, summarise, translate, and reason in natural language. Examples: GPT-4, Llama 3, Gemini. |
| Embedding | — | A list of numbers (a vector) that represents the semantic meaning of a piece of text. Two sentences about the same topic will have embeddings that are mathematically close to each other. |
| Vector | — | Just a list of numbers in a specific order. Embeddings ARE vectors. A 384-dimensional embedding is a list of 384 numbers. |
| Bi-encoder | — | A model architecture where the query and the document are embedded SEPARATELY and then compared. Fast to search (pre-compute document embeddings). Used in the embedding step. BAAI/bge-small-en-v1.5 is a bi-encoder. |
| Cross-encoder | — | A model architecture where the query and document are fed TOGETHER and the model outputs a single relevance score. Much more accurate than bi-encoder but too slow to use on thousands of docs. Used for reranking a small shortlist. |
| Reranking | — | A second-pass scoring step. After vector search gives you the top 20 candidates, a reranker (cross-encoder) re-scores just those 20 for the query and re-sorts them. More accurate final top-5. |
| Chunking | — | Splitting a long document into smaller pieces before indexing. Necessary because embedding models have a token limit, and because smaller focused chunks retrieve more precisely than embedding whole pages. |
| Chunk overlap | — | When chunking, letting consecutive chunks share N words at their boundary. Prevents a sentence split across chunks from being lost. Set to 100 words in this project. |
| Dense retrieval | — | Searching by embedding similarity (vectors). Finds documents that are semantically similar even if they use different words. |
| Sparse retrieval | — | Searching by exact keyword matching (like BM25). Finds documents that share exact words or terms with the query. |
| Hybrid retrieval | — | Combining dense + sparse results. Each method catches what the other misses. Standard practice in production RAG systems. |
| RRF | Reciprocal Rank Fusion | A formula for merging two ranked lists. Each result gets a score of 1/(k + rank), where k=60 is a smoothing constant. Simple and effective — no need to tune weights. |
| Hallucination | — | When an LLM confidently states something that is not true or not supported by the provided context. The refusal mechanism in DocSage explicitly fights this. |
| Guardrail | — | A rule or mechanism that constrains model behaviour. Here: the model is instructed to refuse to answer if the context doesn't support an answer. |
| Fine-tuning | — | Further training a pre-trained model on a specific dataset to improve performance on a specific task. Listed as a future scope item. |
| LoRA | Low-Rank Adaptation | A parameter-efficient fine-tuning technique. Trains only a small number of adapter weights instead of the full model. Feasible on consumer hardware for 7B models. |
| RLHF | Reinforcement Learning from Human Feedback | The training technique used to align LLMs with human preferences. Sunny does annotation tasks that feed into RLHF pipelines at Innodata. |
| ReAct | Reason + Act | An agent pattern where the model alternates between reasoning about what to do next and taking an action (calling a tool). LangGraph implements this as a loop. |
| p50 / p95 latency | 50th/95th percentile latency | p50 is the median response time. p95 means 95% of requests were faster than this. Standard production performance metrics. |
| Quantization | — | Reducing the numerical precision of model weights (e.g., from 32-bit float to 8-bit int) to make them smaller and faster. int8 quantization roughly halves RAM usage. |
| ONNX | Open Neural Network Exchange | A standard file format for ML models. sentence-transformers can export to ONNX, then run with ONNXRuntime which is faster and uses less RAM than PyTorch. |
| Tokenisation | — | Splitting text into tokens (word pieces) before feeding it to an LLM. Tokens ≠ words; "unbelievable" might be 3 tokens. |
| Context window | — | The maximum amount of text an LLM can process in one call. Measured in tokens. Llama 3.3 70B has a 128K token context window. |
| System prompt | — | An instruction given to the LLM before the conversation starts, defining its role and constraints. DocSage's system prompt forces citation format and the refusal phrase. |
| Base64 | — | An encoding that converts binary data (like an image file) into a plain-text string. Used to send images to Groq's vision API inside a JSON payload. |
| Multimodal | — | A model or system that can process more than one type of data (text, images, audio). DocSage is multimodal: it handles text chunks AND image captions. |

---

## RAGAS Evaluation Metrics

| Metric | What it measures |
|---|---|
| Faithfulness | Are all claims in the answer actually supported by the retrieved context? Score of 1.0 = everything said is grounded in the source chunks. |
| Answer Relevancy | Does the answer actually address the question asked? Penalises answers that are technically true but off-topic. |
| Context Precision | Of the retrieved chunks, what fraction were actually useful? High precision = retrieval is not pulling in noise. |
| Context Recall | Did the retrieval find all the chunks needed to answer the question? High recall = no important context was missed. |

---

## Models Used

| Model ID | Type | Used For | Where It Runs |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | Bi-encoder embedding model | Converting text chunks to vectors | Locally (CPU) |
| `meta-llama/llama-4-scout-17b-16e-instruct` | Vision-language model | Captioning architecture diagrams from PDFs | Groq API (free) |
| `llama-3.3-70b-versatile` | Large language model | Answering questions, agent reasoning | Groq API (free) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker | Re-scoring top 20 retrieved chunks | Locally (CPU) |

---

## Libraries and Frameworks

| Library | What it does in this project |
|---|---|
| `pymupdf` (import: `fitz`) | Extracts text and images from PDF files page by page |
| `pdfplumber` | Extracts tables from PDFs as structured rows/cells |
| `sentence-transformers` | Loads the bge-small embedding model and encodes text |
| `chromadb` | The vector database. Stores embeddings + metadata. Persistent on disk. |
| `rank_bm25` | Implements BM25Okapi keyword search |
| `groq` | Official Groq Python SDK for calling Llama models via API |
| `langchain-groq` | LangChain wrapper for Groq, used by LangGraph's agent |
| `langchain-core` | Core abstractions: ChatPromptTemplate, HumanMessage, etc. |
| `langgraph` | Builds the ReAct agent graph. `create_react_agent` is the entry point. |
| `mcp` | Official MCP Python SDK. `FastMCP` for server, `ClientSession` for client. |
| `fastmcp` | High-level MCP server builder (wraps `mcp` with decorators) |
| `ragas` | Evaluation library. Scores faithfulness, relevancy, precision, recall. |
| `datasets` | HuggingFace datasets library. RAGAS uses it internally for eval datasets. |
| `fastapi` | Builds the REST API with automatic OpenAPI docs at `/docs` |
| `uvicorn` | ASGI server that runs the FastAPI app |
| `streamlit` | Builds the web chat frontend with minimal code |
| `python-dotenv` | Reads `.env` file and loads it as environment variables |
| `pydantic` | Data validation. FastAPI uses it for request/response models. Also used to dynamically build MCP tool schemas. |
| `pickle` | Python standard library. Used to save/load the BM25 index to disk. |
| `gc` | Python standard library. `gc.collect()` forces garbage collection to free RAM between PDFs. |

---

## Architecture / System Terms

| Term | Meaning |
|---|---|
| MCP | Model Context Protocol — an open protocol (by Anthropic) that standardises how AI models connect to external tools and data sources. Like USB-C, but for AI tool integration. |
| FastMCP | A Python library that makes writing MCP servers easy using decorators (`@mcp.tool()`). |
| stdio transport | The MCP communication method used here: the client starts the server as a subprocess and communicates via stdin/stdout pipes. Simple, no network port needed. |
| Agent | An LLM that decides which tool to call (or whether to call any), calls it, observes the result, and loops until it has enough to answer. Different from a simple chain that always runs the same steps. |
| LangGraph | A library for building stateful, graph-structured agent workflows. The graph has nodes (LLM call, tool call) and edges (conditions for what to do next). |
| CompiledStateGraph | The object returned by `create_react_agent(...)`. You call `.invoke({"messages": [...]})` on it to run the agent. |
| Tool | In LangGraph context: a Python function wrapped with metadata (name, description, argument schema) so the LLM can decide to call it. |
| StructuredTool | A LangChain class for creating tools with explicit Pydantic schemas. Used to wrap MCP tools for LangGraph. |
| `args_schema` | The Pydantic model that defines what arguments a StructuredTool accepts. Must be set explicitly when wrapping MCP tools (the default auto-inference fails for dynamically created tools). |
| FastAPI | A modern Python web framework for building REST APIs. Faster than Flask. Built-in OpenAPI docs. |
| ASGI | Asynchronous Server Gateway Interface — the interface FastAPI and Uvicorn use to handle async HTTP. |
| Uvicorn | The server that runs FastAPI apps. Equivalent to Gunicorn for WSGI apps. |
| Pydantic model | A Python class that validates and types-checks its fields automatically. Used for request/response bodies in FastAPI. |
| Endpoint | A URL + HTTP method combination that your API exposes. E.g., `POST /query` accepts a question and returns an answer. |
| HF Spaces | Hugging Face Spaces — free cloud hosting for ML demos. Supports Streamlit, Gradio, and Docker SDKs. Your DocSage deployment will use the Docker SDK. |
| Docker SDK (HF) | The HF Spaces deployment mode where you provide a `Dockerfile`. HF builds and runs it server-side. You do NOT run Docker locally. |
| Dockerfile | A text file of instructions for building a Docker container image. Our Dockerfile: installs requirements, exposes port 7860, runs Streamlit. |
| `EXPOSE 7860` | The port HF Spaces expects all Docker-based apps to listen on. Must match the Streamlit port. |
| Space secret | An environment variable you set via the HF Spaces UI settings. At runtime, it becomes a regular env var readable with `os.environ["KEY"]`. Used for `GROQ_API_KEY`. |
| `.env` file | A local file containing key=value pairs (API keys, paths). Loaded by `python-dotenv`. Never commit this to Git. |
| `.env.example` | A template `.env` file with placeholder values. Safe to commit. |
| `gitignore` | A file telling Git which files/folders to exclude from version control. `.env`, `venv/`, `__pycache__/`, raw PDFs. |
| `data/processed/` | The folder where Chroma DB and BM25 index are saved. NOT in gitignore — must be committed so the HF Spaces deployment has the pre-built index. |
| Persistent Chroma client | `chromadb.PersistentClient(path=...)` — saves the vector index to disk so you don't re-embed on every restart. |
| `gc.collect()` | Forces Python's garbage collector to run immediately, reclaiming memory from objects no longer in use. Called between PDF processing to free RAM. |

---

## File and Folder Names

| Path | What it is |
|---|---|
| `src/config.py` | Central settings file. All paths, model names, and hyperparameters. Change once, affects everything. |
| `src/ingestion/extract.py` | Extracts text, tables, and images from PDF files. |
| `src/ingestion/chunk.py` | Splits extracted text into overlapping chunks. |
| `src/embeddings/embed.py` | Turns text chunks into embedding vectors. |
| `src/embeddings/caption.py` | Sends images to Groq vision and gets text descriptions back. |
| `src/retrieval/vector_store.py` | Adds chunks to Chroma; queries Chroma by embedding. |
| `src/retrieval/sparse.py` | Builds and queries BM25 keyword index. |
| `src/retrieval/hybrid.py` | Merges dense + sparse results using RRF. |
| `src/retrieval/rerank.py` | Re-scores merged results with the cross-encoder. |
| `src/retrieval/search.py` | Public entry point: `search_documents(query)` calls hybrid → rerank. |
| `src/generation/generate.py` | Calls Groq LLM with context + query, returns cited answer. |
| `src/agent/mcp_server.py` | FastMCP server exposing `search_documentation` and `estimate_monthly_cost` tools. |
| `src/agent/mcp_client_tools.py` | Wraps MCP tools as LangGraph-compatible StructuredTools. |
| `src/agent/graph.py` | Builds the LangGraph ReAct agent. |
| `src/evaluation/eval_set.json` | 30 Q&A pairs for automated evaluation. |
| `src/evaluation/run_eval.py` | Runs evaluation, prints RAGAS scores + latency metrics. |
| `src/api/main.py` | FastAPI app: `/health` and `/query` endpoints. |
| `app.py` | Streamlit frontend. Entry point for the live demo. |
| `scripts/build_index.py` | Orchestrates the full ingestion pipeline: PDF → index. Run once. |
| `requirements.txt` | Python dependencies. Used by pip and the Dockerfile. |
| `.env.example` | Template for the `.env` file. |
| `.gitignore` | Files Git should ignore. |
| `Dockerfile` | Container build instructions for HF Spaces deployment. |
| `docs/PROJECT_PLAN.md` | This project's overview, rationale, and roadmap. |
| `docs/GLOSSARY.md` | This file. |
| `docs/SETUP_AND_RUN.md` | How to set up the environment and run the project locally. |
| `docs/DEPLOYMENT.md` | How to deploy to HF Spaces and get the resume link. |

---

## Common API / Config Keys

| Key | What it is |
|---|---|
| `GROQ_API_KEY` | Your personal API key from console.groq.com. Never commit. |
| `GENERATION_MODEL` | The Groq model ID used for answering questions (`llama-3.3-70b-versatile`). |
| `VISION_MODEL` | The Groq model ID used for image captioning (`meta-llama/llama-4-scout-17b-16e-instruct`). |
| `EMBEDDING_MODEL` | The HuggingFace model ID for embeddings (`BAAI/bge-small-en-v1.5`). |
| `RERANK_MODEL` | The HuggingFace model ID for reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`). |
| `CHUNK_SIZE` | Target word count per chunk (500). |
| `CHUNK_OVERLAP` | Words shared between consecutive chunks (100). |
| `TOP_K_DENSE` | How many results to fetch from Chroma (10). |
| `TOP_K_SPARSE` | How many results to fetch from BM25 (10). |
| `TOP_K_FINAL` | How many results to pass to the LLM after reranking (5). |
| `EMBED_BATCH_SIZE` | How many chunks to embed at once (8). Low value = less RAM usage. |

# DocSage — Setup and Run Guide

Follow this top to bottom, in order. Every step matters.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10 or 3.11 | 3.12 works but some libs are slower to release wheels for it |
| VS Code | Install the Python extension |
| Git | `git --version` to check |
| A Groq account | Free. Sign up at https://console.groq.com — no credit card required |
| 4GB+ RAM | Read the low-RAM tips in each section |

---

## Step 1: Clone or Create the Repository

If you're starting from the zip file:
```bash
# Unzip the project, then:
cd docsage
git init
git add .
git commit -m "initial commit: DocSage project scaffold"
```

Then create a new repo on GitHub named `docsage` and push:
```bash
git remote add origin https://github.com/YOUR_USERNAME/docsage.git
git branch -M main
git push -u origin main
```

---

## Step 2: Create and Activate Virtual Environment

```bash
# In the docsage/ root folder:
python -m venv venv

# Activate (Windows):
venv\Scripts\activate

# Activate (Mac/Linux):
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt. **Always activate the venv before working on this project.**

In VS Code: `Ctrl+Shift+P` → "Python: Select Interpreter" → choose the one inside `./venv`.

---

## Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**This will take 5-10 minutes** the first time. sentence-transformers downloads PyTorch.

**Low-RAM tip:** If pip runs out of memory during torch installation:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Step 4: Get Your Groq API Key

1. Go to https://console.groq.com
2. Sign in / create account (free, no card)
3. Click "API Keys" in the left sidebar
4. Click "Create API Key"
5. Copy the key — you won't see it again

**Important:** Before you use DocSage, check that the model IDs in `src/config.py` are still current. Groq occasionally retires model versions. The current list is always at: https://console.groq.com/docs/models

---

## Step 5: Configure Environment Variables

```bash
# Copy the example file:
cp .env.example .env
```

Open `.env` and fill in your key:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

**Never commit `.env` to Git.** It's already in `.gitignore`. Double-check: `git status` should never show `.env` as a tracked file.

---

## Step 6: Add Your PDF Documents

Put your AWS documentation PDFs into:
```
data/raw/
```

Good starting docs (all free to download from aws.amazon.com):
- AWS Well-Architected Framework whitepaper
- AWS Well-Architected Tool documentation
- Any AWS service user guide (EC2, S3, Lambda, etc.)

**Start with 1-2 PDFs** while testing. More PDFs = longer indexing time + more RAM.

**Low-RAM tip:** If a PDF is very large (100+ pages), process it alone first, then add more.

---

## Step 7: Build the Index

This is the offline step that reads your PDFs and creates the searchable index.

```bash
python scripts/build_index.py
```

What this does, in order:
1. Scans `data/raw/` for all `.pdf` files
2. For each PDF: extracts text, tables, images
3. Splits text into chunks (500 words, 100-word overlap)
4. Sends any embedded images to Groq vision for captioning
5. Embeds all chunks and image captions using bge-small (CPU)
6. Stores embeddings in Chroma at `data/processed/chroma_db/`
7. Builds BM25 index and saves to `data/processed/bm25_index.pkl`

**Expected time:** ~2-5 minutes per PDF on a 4GB RAM laptop. The embedding step is the bottleneck.

**Low-RAM tip:** If it crashes mid-way, it's safe to re-run — `get_or_create_collection` means Chroma won't duplicate existing chunks if you handle IDs correctly.

After indexing, commit `data/processed/` to Git:
```bash
git add data/processed/
git commit -m "add pre-built index for HF Spaces deployment"
git push
```

**Why commit the index?** Because your HF Spaces deployment won't run `build_index.py` — it just serves the app. Recruiters testing your live demo need the index to already be there. This is the opposite of the usual "don't commit generated files" rule, but it's the right call here.

---

## Step 8: Run the Evaluation (Optional but Recommended)

Before serving the app, check that the pipeline actually works:

```bash
python src/evaluation/run_eval.py
```

This runs 30 questions through the full pipeline and prints:
- RAGAS faithfulness score
- RAGAS answer_relevancy score
- RAGAS context_precision score
- RAGAS context_recall score
- p50 latency (median response time)
- p95 latency (95th percentile response time)

**Troubleshooting — ragas ImportError:**
If you see:
```
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```
This is a known version mismatch between ragas and langchain_community. Fix:
```bash
pip install "ragas==0.1.21" "langchain-community==0.2.16"
```
Check ragas's GitHub Issues for the current recommended version pair if this persists: https://github.com/explodinggradients/ragas/issues

---

## Step 9: Run the FastAPI Backend

```bash
uvicorn src.api.main:app --reload --port 8000
```

Test it's working:
```bash
curl http://localhost:8000/health
# Should return: {"status": "ok"}

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the six pillars of the AWS Well-Architected Framework?"}'
```

Or visit http://localhost:8000/docs in your browser — FastAPI auto-generates interactive API documentation. You can test queries directly from the browser.

---

## Step 10: Run the Streamlit Frontend

Open a **second terminal** (keep uvicorn running in the first), activate the venv, then:

```bash
streamlit run app.py
```

This opens the chat interface at http://localhost:8501

Features in the UI:
- Type a question and press Enter
- The answer appears with source citations [document, page X]
- A toggle for "Agent Mode" enables MCP tool-use (cost calculations, etc.)
- Sources are shown in an expandable section below each answer

---

## Step 11: Run the MCP Server (for development/testing)

The MCP server runs as a subprocess automatically when the agent is used. But you can run it standalone for testing:

```bash
python -m src.agent.mcp_server
```

It will listen on stdio. Test it by running the agent graph directly:

```python
# Quick test in Python REPL
from src.agent.graph import get_agent
agent = get_agent()
result = agent.invoke({"messages": [{"role": "user", "content": "What is the reliability pillar?"}]})
print(result["messages"][-1].content)
```

---

## Development Workflow

When you make changes:

1. Edit the relevant `src/` file
2. If you changed ingestion/embedding code: re-run `scripts/build_index.py`
3. If you changed generation/retrieval code: re-run `src/evaluation/run_eval.py` to check if quality improved or regressed
4. If quality improved: commit and push
5. The HF Spaces deployment auto-rebuilds on push (if you set up the HF remote — see DEPLOYMENT.md)

**Git commit message convention:**
```
feat: add sparse retrieval with BM25
fix: correct chunk overlap calculation
eval: faithfulness improved from 0.81 to 0.87
docs: update setup instructions
```

---

## Project Structure Reference

```
docsage/
├── app.py                    ← Streamlit frontend (entry point)
├── Dockerfile                ← HF Spaces deployment config
├── requirements.txt          ← Python dependencies
├── .env.example              ← Template for API keys
├── .gitignore
├── data/
│   ├── raw/                  ← Put your PDFs here
│   └── processed/            ← Chroma DB + BM25 index (commit this!)
│       ├── chroma_db/
│       ├── bm25_index.pkl
│       └── images/           ← Extracted diagram images
├── docs/
│   ├── PROJECT_PLAN.md
│   ├── GLOSSARY.md
│   ├── SETUP_AND_RUN.md      ← This file
│   └── DEPLOYMENT.md
├── scripts/
│   └── build_index.py        ← Run once to build the index
└── src/
    ├── config.py             ← All settings in one place
    ├── ingestion/
    │   ├── extract.py
    │   └── chunk.py
    ├── embeddings/
    │   ├── embed.py
    │   └── caption.py
    ├── retrieval/
    │   ├── vector_store.py
    │   ├── sparse.py
    │   ├── hybrid.py
    │   ├── rerank.py
    │   └── search.py
    ├── generation/
    │   └── generate.py
    ├── agent/
    │   ├── mcp_server.py
    │   ├── mcp_client_tools.py
    │   └── graph.py
    ├── evaluation/
    │   ├── eval_set.json
    │   └── run_eval.py
    └── api/
        └── main.py
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `GROQ_API_KEY not set` | `.env` not loaded or key missing | Check `.env` exists and has the key; make sure you're running from the project root |
| `No module named 'sentence_transformers'` | Not installed | `pip install sentence-transformers` |
| `ModuleNotFoundError: langchain_community.chat_models.vertexai` | ragas/langchain version mismatch | Pin versions as described in Step 8 |
| `Collection already exists` in Chroma | Re-running build_index without clearing old data | This is fine — `get_or_create_collection` is safe |
| `RateLimitError` from Groq | Hit free tier limits (30 RPM / 1000 RPD) | Wait a minute, or reduce the number of eval questions run in one batch |
| `MemoryError` during embedding | Too many chunks loaded at once | Reduce `EMBED_BATCH_SIZE` in `config.py` to 4 |
| Streamlit shows blank page | FastAPI backend not running | Start uvicorn first, then streamlit |
| `create_react_agent` deprecation warning | LangGraph V1 → V2 transition | Safe to ignore for now; a comment in graph.py notes the future migration path |

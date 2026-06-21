# DocSage — Deployment Guide

How to get DocSage running as a public live link you can put on your resume.

---

## Platform: Hugging Face Spaces

**Why HF Spaces:**
- Free, no credit card required
- Gives you a public URL: `https://huggingface.co/spaces/YOUR_USERNAME/docsage`
- Recruiters can click it and test the app without installing anything
- Rebuilds automatically when you push code

**Important — Docker SDK, not Streamlit SDK:**
HF Spaces has a "Streamlit" option, but their native Streamlit SDK is now deprecated. Use the **Docker SDK** instead — you provide a Dockerfile, they build and run it. This is more reliable and gives you full control.

---

## Step 1: Create a Hugging Face Account

Go to https://huggingface.co and sign up. Free.

---

## Step 2: Create a New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in:
   - **Owner:** your HF username
   - **Space name:** `docsage`
   - **License:** MIT
   - **SDK:** choose **Docker** (not Streamlit, not Gradio)
   - **Docker template:** blank
4. Click "Create Space"

This creates an empty Git repo at:
```
https://huggingface.co/spaces/YOUR_USERNAME/docsage
```

---

## Step 3: Set Your Groq API Key as a Space Secret

Your `GROQ_API_KEY` must be available to the app at runtime, but must NOT be committed to the Git repo.

1. Go to your Space page on HF
2. Click "Settings" tab
3. Scroll to "Repository secrets"
4. Click "New secret"
5. Name: `GROQ_API_KEY`
6. Value: your actual Groq API key (from console.groq.com)
7. Click "Add secret"

**How it works:** In Docker SDK spaces, secrets become plain environment variables. Your Python code reads them with `os.environ["GROQ_API_KEY"]`. The `python-dotenv` call in the app handles both local (reads `.env`) and production (reads env var) automatically — `load_dotenv()` won't overwrite an env var that's already set.

---

## Step 4: Add the HF Space as a Git Remote

In your local docsage folder:

```bash
# Install the HF hub CLI (if not installed)
pip install huggingface_hub

# Login (enter your HF token when prompted)
huggingface-cli login

# Add the Space as a remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/docsage

# Verify remotes
git remote -v
# Should show:
# origin  https://github.com/YOUR_USERNAME/docsage.git
# hf      https://huggingface.co/spaces/YOUR_USERNAME/docsage
```

You now have two remotes: GitHub (for code/portfolio) and HF (for the live demo).

---

## Step 5: Make Sure the Pre-Built Index Is Committed

The HF Space will NOT run `build_index.py`. It just serves `app.py`. Recruiters who visit the live link need the index to already exist.

Check that `data/processed/` contains your built index and is committed:
```bash
ls data/processed/
# Should show: chroma_db/  bm25_index.pkl  images/

git status
# data/processed/ should NOT show as untracked
```

If not committed yet:
```bash
git add data/processed/
git commit -m "add pre-built index for deployment"
```

**Note on `.gitignore`:** The `.gitignore` excludes `data/raw/*.pdf` (your raw PDFs, which can be large) but explicitly does NOT exclude `data/processed/`. This is intentional and opposite to the usual advice of not committing generated files. Document this decision in your repo README so interviewers don't flag it as a mistake.

**Note on file size:** If `chroma_db/` is larger than 100MB, Git will warn you. For large corpora, you'd use Git LFS. For a demo with 2-3 AWS whitepapers, it should be well under 50MB.

---

## Step 6: Push to HF Spaces

```bash
git push hf main
```

HF will build the Docker image. Watch the build logs on your Space page (click "Factory rebuild" if you don't see activity).

Build takes 3-5 minutes the first time (installing requirements). Subsequent pushes are faster due to Docker layer caching.

---

## Step 7: Verify the Deployment

1. Wait for the Space status to show "Running" (green dot)
2. Open the Space URL in an **incognito/private browser window**
3. Ask a question about AWS, e.g.: *"What are the six pillars of the Well-Architected Framework?"*
4. Verify:
   - Answer appears within 5-10 seconds
   - Source citations are shown
   - The "I don't have enough information" refusal works for out-of-scope questions

If the Space shows "Error", click "Logs" to see what failed. Common causes:
- `GROQ_API_KEY` secret not set (most common)
- `data/processed/` not committed
- Python dependency conflict in requirements.txt

---

## Step 8: Add the Live Link to Your Resume

In your resume, under Projects:

```
DocSage — Multimodal RAG Agent for AWS Documentation
Tech: Python, LangGraph, FastMCP, Groq, Chroma, RAGAS, FastAPI, Streamlit, HF Spaces
• Built hybrid retrieval (BM25 + vector search) with cross-encoder reranking
• Integrated MCP tool-use layer via LangGraph ReAct agent
• Implemented RAGAS evaluation harness (faithfulness, relevancy, precision/recall metrics)
• Handles multimodal inputs: text, tables, and architecture diagrams (Llama 4 Scout captioning)

Live demo: https://huggingface.co/spaces/YOUR_USERNAME/docsage
Code: https://github.com/YOUR_USERNAME/docsage
```

Also add the live demo link in your GitHub repo's "About" section (the box on the right side of the repo page).

---

## Keeping Both Remotes in Sync

When you make improvements locally:
```bash
# 1. Commit your changes
git add .
git commit -m "feat: improve reranking threshold"

# 2. Push to GitHub (portfolio)
git push origin main

# 3. Push to HF Spaces (live demo)
git push hf main
```

---

## Troubleshooting Deployment

| Symptom | Likely Cause | Fix |
|---|---|---|
| Space shows "Building" forever | Large requirements.txt | Slim requirements.txt; move dev dependencies out |
| `Exec format error` | Wrong base image arch | Use `FROM python:3.11-slim` not platform-specific images |
| App loads but gives errors | Missing Space secret | Check Settings → Secrets for `GROQ_API_KEY` |
| `No such file: data/processed/chroma_db` | Index not committed | `git add data/processed/ && git commit && git push hf main` |
| App works locally but not on HF | Absolute paths in code | Check config.py — all paths must be relative or use `Path(__file__).parent` |
| Groq API errors in prod | Rate limit hit | Free tier: 30 RPM, 1000 RPD. Consider adding a retry with backoff. |

---

## Space Embed (Optional — for LinkedIn or Portfolio Website)

HF Spaces gives you an embed code. From your Space page → "..." menu → "Embed this Space". You can paste an iframe into your portfolio website.

---

## Dockerfile Reference

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

Key points:
- Port MUST be 7860 — that's what HF Spaces routes to
- `--server.headless=true` prevents Streamlit from trying to open a browser
- `--server.address=0.0.0.0` makes it reachable from outside the container

"""Central configuration. Every other module reads settings from here
instead of calling os.environ directly, so there is exactly one place
to change a model name, a path, or a default."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
IMAGE_DIR = DATA_PROCESSED_DIR / "images"
CHROMA_DIR = DATA_PROCESSED_DIR / "chroma_db"
BM25_INDEX_PATH = DATA_PROCESSED_DIR / "bm25_index.pkl"

# --- API keys ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# --- Model names ---
# Groq renames/retires models periodically. If a call fails with a
# "model not found" error, check the current list at
# https://console.groq.com/docs/models and update these two lines.
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "llama-3.3-70b-versatile")
VISION_MODEL = os.environ.get("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# Small, CPU-friendly models chosen specifically for low-RAM machines.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Retrieval tuning ---
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))       # words per text chunk
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 100))  # word overlap between chunks
TOP_K_DENSE = int(os.environ.get("TOP_K_DENSE", 10))
TOP_K_SPARSE = int(os.environ.get("TOP_K_SPARSE", 10))
TOP_K_FINAL = int(os.environ.get("TOP_K_FINAL", 5))
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", 8))  # kept small for 4GB RAM

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, IMAGE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

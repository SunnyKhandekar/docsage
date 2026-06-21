"""
scripts/build_index.py
----------------------
Orchestrates the full ingestion pipeline: PDFs → searchable index.

Run this once before starting the API or Streamlit app.
After running, commit data/processed/ to Git for the HF Spaces deployment.

Pipeline per PDF:
  1. extract_pdf()    — text, tables, images from each page
  2. chunk_page()     — split text into 500-word overlapping chunks
  3. caption_image()  — Groq vision describes each extracted image
  4. embed_texts()    — sentence-transformers encodes all chunk texts
  5. add_chunks()     — store vectors + metadata in Chroma
  After all PDFs: build_bm25_index() — build keyword index over all chunks

Low-RAM strategy:
  Process one PDF at a time. Call gc.collect() between PDFs. Use small
  embedding batch sizes (EMBED_BATCH_SIZE=8 in config.py). Never load all
  PDFs into memory simultaneously.
"""

import gc
import logging
import sys
from pathlib import Path

# Make sure src/ is importable when running from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.ingestion.extract import extract_pdf
from src.ingestion.chunk import chunk_page
from src.embeddings.embed import embed_texts
from src.embeddings.caption import caption_image
from src.retrieval.vector_store import add_chunks
from src.retrieval.sparse import build_bm25_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("build_index")


def process_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract, chunk, and caption one PDF. Returns list of chunk dicts ready
    for embedding and indexing.
    """
    logger.info(f"Processing: {pdf_path.name}")
    pages = extract_pdf(pdf_path)

    all_chunks = []

    for page in pages:
        # Text chunks
        text_chunks = chunk_page(page, source_doc=pdf_path.name)
        all_chunks.extend([c.to_dict() for c in text_chunks])

        # Image captions — each image becomes a chunk of type "image_caption"
        for img_path in page.image_paths:
            try:
                caption = caption_image(img_path)
                caption_chunk = {
                    "chunk_id": f"{pdf_path.stem}_page{page.page_num}_img_{img_path.stem}",
                    "text": caption,
                    "source_doc": pdf_path.name,
                    "page_num": page.page_num,
                    "chunk_type": "image_caption",
                }
                all_chunks.append(caption_chunk)
            except Exception as e:
                logger.warning(f"Caption failed for {img_path}: {e}")

    logger.info(f"  → {len(all_chunks)} chunks from {pdf_path.name}")
    return all_chunks


def main():
    pdf_files = sorted(RAW_DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        logger.error(
            f"No PDF files found in {RAW_DATA_DIR}. "
            "Add AWS documentation PDFs and re-run."
        )
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDF(s): {[p.name for p in pdf_files]}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks_for_bm25 = []

    for pdf_path in pdf_files:
        # Process one PDF at a time to keep memory low.
        chunks = process_pdf(pdf_path)

        # Embed and store in Chroma.
        texts = [c["text"] for c in chunks]
        logger.info(f"  Embedding {len(texts)} chunks (batch_size from config)...")
        embeddings = embed_texts(texts)

        logger.info(f"  Adding to Chroma ...")
        add_chunks(chunks, embeddings)

        all_chunks_for_bm25.extend(chunks)

        # Free memory before next PDF.
        del chunks, texts, embeddings
        gc.collect()
        logger.info(f"  Done. Memory freed.")

    # Build BM25 index over the entire corpus.
    logger.info(f"Building BM25 index over {len(all_chunks_for_bm25)} total chunks ...")
    build_bm25_index(all_chunks_for_bm25)

    logger.info("\n✅ Index build complete.")
    logger.info(f"   Chroma DB: {PROCESSED_DATA_DIR / 'chroma_db'}")
    logger.info(f"   BM25 index: {PROCESSED_DATA_DIR / 'bm25_index.pkl'}")
    logger.info("\nNext steps:")
    logger.info("  1. git add data/processed/ && git commit -m 'add pre-built index'")
    logger.info("  2. uvicorn src.api.main:app --reload --port 8000")
    logger.info("  3. streamlit run app.py")


if __name__ == "__main__":
    main()

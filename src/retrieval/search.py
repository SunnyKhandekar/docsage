"""
src/retrieval/search.py
-----------------------
Public entry point for document search.

This is the single function that everything else calls:
  - The MCP server tool (search_documentation)
  - The FastAPI endpoint
  - The evaluation harness

It wires together: hybrid_search → rerank → return top chunks with metadata.
"""

import logging
from typing import List, Dict, Any

from src.retrieval.hybrid import hybrid_search
from src.retrieval.rerank import rerank
from src.config import TOP_K_FINAL

logger = logging.getLogger(__name__)


def search_documents(
    query: str,
    top_k: int = TOP_K_FINAL,
) -> List[Dict[str, Any]]:
    """
    Search the indexed documents for chunks relevant to the query.

    Pipeline:
        1. hybrid_search: BM25 + Chroma, merged via RRF
        2. rerank: cross-encoder re-scores the merged candidates

    Args:
        query: Natural language question or search string.
        top_k: Number of chunks to return.

    Returns:
        List of chunk dicts, sorted by relevance (most relevant first).
        Each dict contains:
          - chunk_id: str
          - text: str
          - source_doc: str (filename)
          - page_num: int
          - chunk_type: "text" | "table" | "image_caption"
          - rrf_score: float
          - rerank_score: float
    """
    logger.info(f"search_documents called: query={query!r}, top_k={top_k}")

    candidates = hybrid_search(query)
    results = rerank(query, candidates, top_k=top_k)

    logger.info(
        f"search_documents returning {len(results)} chunks for query: {query!r}"
    )
    return results

"""
src/retrieval/rerank.py
-----------------------
Cross-encoder reranking of retrieved chunks.

Why rerank?
  Embedding (bi-encoder) models embed query and documents SEPARATELY, then
  compare vectors. This is fast but less accurate because the query and
  document never "see" each other during scoring.

  A cross-encoder feeds the query and document TOGETHER and outputs a single
  relevance score. This is much more accurate but too slow to run on thousands
  of docs — so we run it only on the top ~20 candidates from hybrid search.

  Net effect: hybrid search recalls broadly, reranker sharpens the final top-5.

Model choice:
  cross-encoder/ms-marco-MiniLM-L-6-v2 — chosen specifically for low RAM.
  ms-marco models are trained on real search queries, making them well-suited
  for document retrieval. MiniLM-L-6 is the smallest variant (~22MB on disk).
  bge-reranker-base is stronger but uses ~280MB+ RAM. On a 4GB laptop the
  MiniLM variant is the right tradeoff.
"""

import logging
from typing import List, Dict, Any
from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.config import RERANK_MODEL, TOP_K_FINAL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    """
    Load the cross-encoder model once and cache it in memory.
    lru_cache with maxsize=1 acts as a module-level singleton without a global.
    """
    logger.info(f"Loading reranker: {RERANK_MODEL}")
    return CrossEncoder(RERANK_MODEL)


def rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = TOP_K_FINAL,
) -> List[Dict[str, Any]]:
    """
    Re-score a list of chunks against the query using a cross-encoder, and
    return the top_k highest-scoring chunks.

    Args:
        query: The original user query.
        chunks: List of chunk dicts from hybrid search (already filtered).
        top_k: Number of top chunks to return after reranking.

    Returns:
        Top-k chunk dicts, sorted by rerank score descending.
        Each chunk has a "rerank_score" float field added.
    """
    if not chunks:
        return []

    reranker = _get_reranker()

    # CrossEncoder.predict takes a list of [query, passage] pairs.
    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = reranker.predict(pairs)

    # Attach scores and sort.
    scored = sorted(
        zip(scores, chunks),
        key=lambda x: float(x[0]),
        reverse=True,
    )[:top_k]

    results = []
    for score, chunk in scored:
        result = dict(chunk)
        result["rerank_score"] = float(score)
        results.append(result)

    logger.debug(f"Reranker returned top {len(results)} of {len(chunks)} chunks")
    return results

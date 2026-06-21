"""
src/retrieval/hybrid.py
-----------------------
Hybrid retrieval: combines dense (vector) and sparse (BM25) results using
Reciprocal Rank Fusion (RRF).

Why hybrid?
  - Dense search (Chroma): great for semantic similarity, misses exact terms
  - Sparse search (BM25): great for exact keyword matches, misses paraphrases
  - Together: catch what either one alone would miss

RRF formula:
  score(d) = sum over each ranked list: 1 / (k + rank(d))
  where k=60 is a smoothing constant (standard default, usually not tuned).

RRF is parameter-light: no weights to tune, no training required. Each list
contributes equally. A document appearing in both lists gets a combined score
higher than one appearing in only one.
"""

import logging
from typing import List, Dict, Any

from src.retrieval.vector_store import query_dense
from src.retrieval.sparse import query_sparse
from src.config import TOP_K_DENSE, TOP_K_SPARSE

logger = logging.getLogger(__name__)

RRF_K = 60  # Standard smoothing constant for RRF


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    id_key: str = "chunk_id",
    k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked lists of chunks using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each element is a list of chunk dicts, already sorted
                      by relevance (most relevant first).
        id_key: The dict key to use as the unique chunk identifier.
        k: RRF smoothing constant.

    Returns:
        A single merged list of chunk dicts, sorted by RRF score descending.
        Each returned chunk includes an "rrf_score" field.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_store: Dict[str, Dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            cid = chunk[id_key]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            # Keep the first-seen copy of the chunk metadata.
            if cid not in chunk_store:
                chunk_store[cid] = chunk

    merged = []
    for cid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        result = dict(chunk_store[cid])
        result["rrf_score"] = score
        merged.append(result)

    return merged


def hybrid_search(
    query: str,
    top_k_dense: int = TOP_K_DENSE,
    top_k_sparse: int = TOP_K_SPARSE,
) -> List[Dict[str, Any]]:
    """
    Run both dense and sparse retrieval, merge results with RRF.

    Args:
        query: Natural language query string.
        top_k_dense: How many results to fetch from Chroma.
        top_k_sparse: How many results to fetch from BM25.

    Returns:
        Merged list of chunk dicts sorted by RRF score descending.
    """
    logger.info(f"Hybrid search for: {query!r}")

    dense_results = query_dense(query, top_k=top_k_dense)
    sparse_results = query_sparse(query, top_k=top_k_sparse)

    logger.debug(f"Dense: {len(dense_results)} results, Sparse: {len(sparse_results)} results")

    merged = reciprocal_rank_fusion([dense_results, sparse_results])
    logger.debug(f"Merged: {len(merged)} unique results after RRF")

    return merged

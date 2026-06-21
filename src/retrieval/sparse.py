"""
src/retrieval/sparse.py
-----------------------
BM25 keyword-based (sparse) retrieval.

BM25 (Best Match 25) is a classic information-retrieval algorithm that scores
documents based on exact term overlap with the query. It's fast, needs no GPU,
and catches exact-match queries that vector search sometimes misses (e.g., a
specific AWS service name like "S3 Intelligent-Tiering").

The index is built once and saved to disk with pickle. On subsequent runs it's
loaded from disk instead of being rebuilt.
"""

import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

from src.config import BM25_INDEX_PATH, TOP_K_SPARSE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_bm25_index(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    """
    Build a BM25 index from a list of chunk dicts.

    Each chunk dict must have a "text" key containing the chunk's text content.
    The index is saved to disk at BM25_INDEX_PATH alongside the corpus metadata.

    Args:
        chunks: List of chunk dicts, each with at least {"text": str, ...}.

    Returns:
        The built BM25Okapi index object.
    """
    logger.info(f"Building BM25 index over {len(chunks)} chunks ...")

    # Tokenise by whitespace — simple but effective for English technical docs.
    tokenised = [chunk["text"].lower().split() for chunk in chunks]

    bm25 = BM25Okapi(tokenised)

    # Persist both the index and the corpus so query_sparse can return full
    # chunk metadata (chunk_id, source doc, page number etc.).
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"bm25": bm25, "chunks": chunks}
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)

    logger.info(f"BM25 index saved to {BM25_INDEX_PATH}")
    return bm25


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def load_bm25_index() -> tuple[BM25Okapi, List[Dict[str, Any]]]:
    """
    Load the BM25 index and corpus from disk.

    Returns:
        Tuple of (BM25Okapi index, list of chunk dicts).

    Raises:
        FileNotFoundError: if build_bm25_index has not been run yet.
    """
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_INDEX_PATH}. "
            "Run scripts/build_index.py first."
        )
    with open(BM25_INDEX_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["bm25"], payload["chunks"]


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------

def query_sparse(query: str, top_k: int = TOP_K_SPARSE) -> List[Dict[str, Any]]:
    """
    Search the BM25 index and return the top-k matching chunks.

    Args:
        query: Natural language query string.
        top_k: Number of results to return.

    Returns:
        List of chunk dicts, each augmented with a "bm25_score" float field.
        Sorted by score descending. May return fewer than top_k if the corpus
        is smaller.
    """
    bm25, chunks = load_bm25_index()

    tokenised_query = query.lower().split()
    scores = bm25.get_scores(tokenised_query)

    # Pair each chunk with its score, sort descending, take top_k.
    scored = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True,
    )[:top_k]

    results = []
    for score, chunk in scored:
        result = dict(chunk)
        result["bm25_score"] = float(score)
        results.append(result)

    logger.debug(f"BM25 returned {len(results)} results for query: {query!r}")
    return results

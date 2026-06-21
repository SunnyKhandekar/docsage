"""Thin wrapper around a local, persistent Chroma collection.

API confirmed against chromadb 1.5.9: PersistentClient(path=...),
get_or_create_collection(name=...), .add(ids=, embeddings=, documents=,
metadatas=), and .query(query_embeddings=, n_results=) returning a dict
of parallel lists keyed by ids/documents/metadatas/distances.
"""
import chromadb

from src.config import CHROMA_DIR
from src.embeddings.embed import embed_texts

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(name="docsage_chunks")
    return _collection


def add_chunks(chunks: list[dict]) -> None:
    """chunks: list of {chunk_id, text, source_doc, page_number, chunk_type}"""
    if not chunks:
        return
    collection = get_collection()
    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "source_doc": c["source_doc"],
            "page_number": c["page_number"],
            "chunk_type": c["chunk_type"],
        }
        for c in chunks
    ]
    embeddings = embed_texts(texts)
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def query_dense(query: str, top_k: int = 10) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    query_embedding = embed_texts([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    if not results["ids"][0]:
        return []
    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "dense_score": 1 - results["distances"][0][i],
        })
    return hits

"""Generate dense embeddings for text chunks with a small, CPU-friendly model.

Low-RAM tip: if this feels slow or memory-hungry on a 4GB machine, switch
to the ONNX backend for roughly 2x the speed and a smaller memory
footprint on the same model:

    SentenceTransformer(EMBEDDING_MODEL, backend="onnx",
                         model_kwargs={"file_name": "onnx/model_qint8_avx512.onnx"})

Check the model's Hugging Face page for the exact quantized file name
available before using this.
"""
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBED_BATCH_SIZE

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _model


def embed_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    """Embed a list of texts in small batches to keep peak memory low."""
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.tolist()

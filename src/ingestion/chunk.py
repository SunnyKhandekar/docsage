"""Chunk extracted page content into retrieval-sized pieces.

Tables are never split mid-row -- a table chunk is always kept whole,
since a half a table is worse than no table.
"""
from dataclasses import dataclass

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.ingestion.extract import ExtractedPage, table_to_text


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_doc: str
    page_number: int
    chunk_type: str  # "text", "table", or "image_caption"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_doc": self.source_doc,
            "page_number": self.page_number,
            "chunk_type": self.chunk_type,
        }


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    pieces = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        pieces.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return pieces


def chunk_page(page: ExtractedPage) -> list[Chunk]:
    """Turn one extracted page into prose chunks plus one chunk per table."""
    chunks = []

    for i, piece in enumerate(_split_text(page.text, CHUNK_SIZE, CHUNK_OVERLAP)):
        if not piece.strip():
            continue
        chunks.append(Chunk(
            chunk_id=f"{page.source_doc}_p{page.page_number}_text{i}",
            text=piece,
            source_doc=page.source_doc,
            page_number=page.page_number,
            chunk_type="text",
        ))

    for t_idx, table in enumerate(page.tables):
        table_text = table_to_text(table)
        if table_text.strip():
            chunks.append(Chunk(
                chunk_id=f"{page.source_doc}_p{page.page_number}_table{t_idx}",
                text=table_text,
                source_doc=page.source_doc,
                page_number=page.page_number,
                chunk_type="table",
            ))

    return chunks

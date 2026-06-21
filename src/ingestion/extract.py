"""Extract text, tables, and images from PDF files.

Deliberately avoids unstructured.io's default "hi_res" layout mode,
which pulls in detectron2 and large layout-detection model weights --
that alone can use more RAM than this whole project should need.
PyMuPDF (text + images) and pdfplumber (tables) are both rule-based,
pure-CPU, and lightweight, which matters on a 4GB machine.
"""
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber


@dataclass
class ExtractedPage:
    source_doc: str
    page_number: int
    text: str
    tables: list = field(default_factory=list)       # list of list-of-rows
    image_paths: list = field(default_factory=list)  # paths to extracted images on disk


def extract_pdf(pdf_path: Path, image_output_dir: Path) -> list[ExtractedPage]:
    """Extract text, tables, and images from a single PDF, page by page."""
    pages = []
    doc_name = pdf_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    fitz_doc = fitz.open(pdf_path)

    with pdfplumber.open(pdf_path) as plumber_doc:
        for page_num in range(len(fitz_doc)):
            fitz_page = fitz_doc[page_num]
            text = fitz_page.get_text("text")

            image_paths = []
            for img_index, img in enumerate(fitz_page.get_images(full=True)):
                xref = img[0]
                base_image = fitz_doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                img_path = image_output_dir / f"{doc_name}_p{page_num + 1}_{img_index}.{ext}"
                img_path.write_bytes(image_bytes)
                image_paths.append(str(img_path))

            tables = []
            if page_num < len(plumber_doc.pages):
                tables = plumber_doc.pages[page_num].extract_tables() or []

            pages.append(ExtractedPage(
                source_doc=doc_name,
                page_number=page_num + 1,
                text=text,
                tables=tables,
                image_paths=image_paths,
            ))

    fitz_doc.close()
    return pages


def table_to_text(table: list) -> str:
    """Flatten a table (list of rows) into a readable text block for chunking."""
    lines = []
    for row in table:
        clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
        lines.append(" | ".join(clean_row))
    return "\n".join(lines)

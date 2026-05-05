import re
from pathlib import Path

from pypdf import PdfReader

from . import store
from .config import settings
from .llm import ollama


_WS_RUN = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS_RUN.sub(" ", text).strip()


def _extract_pages(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    return [_clean(page.extract_text() or "") for page in reader.pages]


def _chunk_page(text: str, size: int, overlap: int) -> list[str]:
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
        start += step
    return chunks


def _build_chunks(pages: list[str], filename: str):
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    for page_idx, page_text in enumerate(pages, start=1):
        page_chunks = _chunk_page(page_text, settings.chunk_size, settings.chunk_overlap)
        for chunk_idx, chunk in enumerate(page_chunks):
            ids.append(f"{filename}::p{page_idx}::c{chunk_idx}")
            docs.append(chunk)
            metas.append(
                {"source": filename, "page": page_idx, "chunk_index": chunk_idx}
            )
    return ids, docs, metas


def _batched(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def ingest_pdf(filename: str, raw_bytes: bytes) -> dict:
    upload_path = Path(settings.upload_dir) / filename
    upload_path.write_bytes(raw_bytes)

    pages = _extract_pages(upload_path)
    ids, docs, metas = _build_chunks(pages, filename)

    if not ids:
        return {
            "filename": filename,
            "pages": len(pages),
            "chunks": 0,
            "note": "no extractable text (scanned PDF?)",
        }

    store.delete_by_source(filename)

    embeddings: list[list[float]] = []
    for batch in _batched(docs, 32):
        embeddings.extend(ollama.embed(batch))

    store.add_chunks(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    return {"filename": filename, "pages": len(pages), "chunks": len(ids)}

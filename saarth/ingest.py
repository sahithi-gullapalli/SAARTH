"""Document loading and chunking.

Uses LangChain's RecursiveCharacterTextSplitter when installed, otherwise an equivalent
built-in recursive splitter. LlamaIndex documents are supported via `from_llamaindex_documents`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from saarth.schemas import Chunk

SUPPORTED = {".txt", ".md", ".pdf"}
_SEPS = ("\n\n", "\n", ". ", " ", "")


def load_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_directory(directory: str | Path) -> list[tuple[str, str]]:
    root = Path(directory)
    if not root.exists():
        return []
    return [(p.name, load_file(p)) for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED]


def _atoms(text: str, size: int, seps: tuple[str, ...]) -> list[str]:
    """Break text into pieces no longer than `size`, using the coarsest separator possible."""
    if len(text) <= size:
        return [text]
    sep = next(s for s in seps if s == "" or s in text)
    if sep == "":
        return [text[i : i + size] for i in range(0, len(text), size)]
    rest = seps[seps.index(sep) + 1 :]
    parts = text.split(sep)
    out: list[str] = []
    for i, p in enumerate(parts):
        piece = p + (sep if i < len(parts) - 1 else "")
        out.extend([piece] if len(piece) <= size else _atoms(piece, size, rest))
    return out


def _builtin_split(text: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for atom in _atoms(text, size, _SEPS):
        if length + len(atom) > size and current:
            chunks.append("".join(current).strip())
            while current and (length > overlap or length + len(atom) > size):
                length -= len(current.pop(0))
        current.append(atom)
        length += len(atom)
    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return _builtin_split(text, chunk_size, chunk_overlap)
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_text(text)


def chunk_document(source: str, text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[Chunk]:
    out = []
    for i, piece in enumerate(split_text(text, chunk_size, chunk_overlap)):
        cid = hashlib.sha1(f"{source}:{i}:{piece}".encode()).hexdigest()[:16]
        out.append(Chunk(id=cid, text=piece, source=source, metadata={"chunk": i}))
    return out


def from_llamaindex_documents(documents: list[Any], chunk_size: int = 800, chunk_overlap: int = 120) -> list[Chunk]:
    """Convert LlamaIndex `Document` objects (from any LlamaIndex reader) into SAARTH chunks."""
    chunks: list[Chunk] = []
    for d in documents:
        source = (getattr(d, "metadata", {}) or {}).get("file_name") or getattr(d, "doc_id", "llamaindex")
        chunks.extend(chunk_document(str(source), d.text, chunk_size, chunk_overlap))
    return chunks

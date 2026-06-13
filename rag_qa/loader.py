"""Document loading and chunking.

Loads ``.txt`` and ``.md`` files (and ``.pdf`` when :mod:`pypdf` is available,
silently skipped otherwise) from a directory, then splits their text into
overlapping word-based chunks suitable for embedding.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    """A loaded source document."""

    source: str  #: file name used as the citation label
    text: str  #: full plain-text content


def load_documents(docs_dir: str) -> List[Document]:
    """Load every supported document under ``docs_dir`` (non-recursive)."""
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"docs directory not found: {docs_dir!r}")

    documents: List[Document] = []
    for name in sorted(os.listdir(docs_dir)):
        path = os.path.join(docs_dir, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in (".txt", ".md"):
            with open(path, "r", encoding="utf-8") as fh:
                documents.append(Document(source=name, text=fh.read()))
        elif ext == ".pdf":
            text = _read_pdf(path)
            if text is not None:
                documents.append(Document(source=name, text=text))
    return documents


def _read_pdf(path: str) -> str | None:
    """Extract text from a PDF, or return ``None`` if pypdf is unavailable."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(
    text: str,
    chunk_size: int = 60,
    overlap: int = 15,
) -> List[str]:
    """Split ``text`` into overlapping word windows.

    Args:
        text: the input text.
        chunk_size: target number of words per chunk.
        overlap: number of words shared between consecutive chunks, which
            preserves context that would otherwise be cut at a boundary.

    Returns:
        A list of chunk strings (empty list for empty input).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if window:
            chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks

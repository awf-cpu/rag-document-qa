"""In-memory vector index using NumPy cosine similarity.

Stores embedded chunks and retrieves the top-k most similar to a query vector.
Deliberately tiny — no FAISS/Chroma required — but the interface mirrors what a
real vector store would expose, so swapping one in later is straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class Chunk:
    """A retrievable unit of text with provenance."""

    source: str  #: originating document, used for citations
    chunk_id: int  #: index of this chunk within its source document
    text: str  #: the chunk's text


class VectorIndex:
    """A minimal cosine-similarity vector index over text chunks."""

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._matrix: np.ndarray | None = None  # shape (n_chunks, dim)

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Add chunks together with their pre-computed embedding vectors."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return

        new = _l2_normalize_rows(np.asarray(vectors, dtype=np.float32))
        self._matrix = new if self._matrix is None else np.vstack([self._matrix, new])
        self._chunks.extend(chunks)

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 3,
    ) -> List[Tuple[Chunk, float]]:
        """Return up to ``top_k`` ``(chunk, score)`` pairs by cosine similarity."""
        if self._matrix is None or not self._chunks:
            return []

        q = _l2_normalize_rows(np.asarray([query_vector], dtype=np.float32))[0]
        scores = self._matrix @ q  # cosine sim because all rows are unit-norm
        k = min(top_k, len(self._chunks))
        # argpartition for the top-k, then sort just those by score desc
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self._chunks[i], float(scores[i])) for i in top_idx]


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms

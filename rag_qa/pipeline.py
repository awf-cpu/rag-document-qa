"""End-to-end RAG pipeline tying the pieces together.

``RagPipeline`` ingests a folder of documents, builds an in-memory vector index,
and answers questions by retrieving the most relevant chunks, constructing a
grounded prompt, and asking an :class:`~rag_qa.providers.LLMProvider` to answer
with citations back to the source chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .index import Chunk, VectorIndex
from .loader import chunk_text, load_documents
from .providers import (
    EmbeddingProvider,
    LLMProvider,
    get_embedding_provider,
    get_llm_provider,
)


@dataclass
class Answer:
    """A grounded answer plus the chunks it was built from."""

    question: str
    text: str
    sources: List[Tuple[Chunk, float]] = field(default_factory=list)

    @property
    def citations(self) -> List[str]:
        """Distinct source labels referenced by this answer, in rank order."""
        seen: List[str] = []
        for chunk, _ in self.sources:
            if chunk.source not in seen:
                seen.append(chunk.source)
        return seen


class RagPipeline:
    """Retrieval-augmented generation over a document folder."""

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        llm: LLMProvider | None = None,
        chunk_size: int = 60,
        overlap: int = 15,
    ) -> None:
        self.embedder = embedder or get_embedding_provider()
        self.llm = llm or get_llm_provider()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.index = VectorIndex()

    # -- ingestion --------------------------------------------------------- #
    def ingest(self, docs_dir: str) -> int:
        """Load, chunk, embed and index every document in ``docs_dir``.

        Returns the number of chunks added to the index.
        """
        documents = load_documents(docs_dir)
        chunks: List[Chunk] = []
        for doc in documents:
            for i, piece in enumerate(chunk_text(doc.text, self.chunk_size, self.overlap)):
                chunks.append(Chunk(source=doc.source, chunk_id=i, text=piece))

        if not chunks:
            return 0

        vectors = self.embedder.embed([c.text for c in chunks])
        self.index.add(chunks, vectors)
        return len(chunks)

    # -- retrieval --------------------------------------------------------- #
    def retrieve(self, question: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the ``top_k`` chunks most relevant to ``question``."""
        query_vec = self.embedder.embed([question])[0]
        return self.index.search(query_vec, top_k=top_k)

    # -- generation -------------------------------------------------------- #
    def answer(self, question: str, top_k: int = 3) -> Answer:
        """Retrieve context and produce a grounded, cited answer."""
        retrieved = self.retrieve(question, top_k=top_k)
        prompt = build_prompt(question, retrieved)
        text = self.llm.complete(prompt)
        return Answer(question=question, text=text, sources=retrieved)


def build_prompt(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """Construct a grounded prompt with one labelled block per retrieved chunk.

    The ``[source#chunk_id]`` labels are what the LLM is instructed to cite, and
    are parsed back out by the offline mock provider.
    """
    context_lines = [
        f"[{chunk.source}#{chunk.chunk_id}] {chunk.text}" for chunk, _ in retrieved
    ]
    context = "\n".join(context_lines) if context_lines else "(no documents retrieved)"
    return (
        "You are a question-answering assistant. Answer ONLY from the context "
        "below and cite each statement with its [source#id] tag.\n\n"
        f"QUESTION: {question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "INSTRUCTIONS: If the context does not contain the answer, say so. "
        "Always include citations.\n\n"
        "ANSWER:"
    )

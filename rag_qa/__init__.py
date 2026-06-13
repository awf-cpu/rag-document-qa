"""rag_qa: a small, dependency-light retrieval-augmented generation Q&A engine.

The package exposes a clean pipeline (load -> chunk -> embed -> index ->
retrieve -> answer) built around two provider abstractions, ``EmbeddingProvider``
and ``LLMProvider``. Both ship with deterministic offline "mock" implementations
so the whole system runs and is testable without any API keys.
"""

from .providers import (
    EmbeddingProvider,
    LLMProvider,
    MockEmbeddingProvider,
    MockLLMProvider,
    get_embedding_provider,
    get_llm_provider,
)
from .index import VectorIndex, Chunk
from .pipeline import RagPipeline, Answer

__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "MockEmbeddingProvider",
    "MockLLMProvider",
    "get_embedding_provider",
    "get_llm_provider",
    "VectorIndex",
    "Chunk",
    "RagPipeline",
    "Answer",
]

__version__ = "0.1.0"

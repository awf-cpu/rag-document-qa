"""Tests for embedding, indexing and retrieval with the offline mock provider."""

import os

from rag_qa.index import Chunk, VectorIndex
from rag_qa.pipeline import RagPipeline
from rag_qa.providers import MockEmbeddingProvider

SAMPLE_DOCS = os.path.join(os.path.dirname(__file__), "..", "sample_docs")


def test_mock_embeddings_are_deterministic_and_normalised():
    embedder = MockEmbeddingProvider(dim=64)
    a = embedder.embed(["the water cycle"])[0]
    b = embedder.embed(["the water cycle"])[0]
    assert a == b  # deterministic
    norm = sum(x * x for x in a) ** 0.5
    assert abs(norm - 1.0) < 1e-6  # L2 normalised


def test_index_search_ranks_similar_chunk_first():
    embedder = MockEmbeddingProvider(dim=128)
    chunks = [
        Chunk("doc", 0, "photosynthesis converts sunlight into glucose"),
        Chunk("doc", 1, "http status codes describe request outcomes"),
        Chunk("doc", 2, "the water cycle moves water through evaporation"),
    ]
    index = VectorIndex()
    index.add(chunks, embedder.embed([c.text for c in chunks]))

    query_vec = embedder.embed(["how does photosynthesis use sunlight"])[0]
    results = index.search(query_vec, top_k=1)
    assert results[0][0].chunk_id == 0


def test_pipeline_retrieves_relevant_document():
    pipeline = RagPipeline(embedder=MockEmbeddingProvider(dim=256))
    n = pipeline.ingest(SAMPLE_DOCS)
    assert n > 0

    results = pipeline.retrieve("what gas do plants release during photosynthesis", top_k=3)
    top_sources = {chunk.source for chunk, _ in results}
    assert "photosynthesis.md" in top_sources


def test_empty_index_returns_no_results():
    index = VectorIndex()
    assert index.search([0.0, 1.0], top_k=3) == []

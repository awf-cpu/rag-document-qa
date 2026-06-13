"""End-to-end test: a full answer contains a citation to a source chunk."""

import os
import re

from rag_qa.pipeline import RagPipeline, build_prompt
from rag_qa.index import Chunk

SAMPLE_DOCS = os.path.join(os.path.dirname(__file__), "..", "sample_docs")
CITATION_RE = re.compile(r"\[[^\]]+#\d+\]")


def test_answer_contains_citation():
    pipeline = RagPipeline()  # defaults to mock embedder + mock LLM
    pipeline.ingest(SAMPLE_DOCS)
    answer = pipeline.answer("How does the water cycle work?", top_k=3)

    assert CITATION_RE.search(answer.text), "answer should contain a [source#id] citation"
    assert answer.citations, "answer should expose at least one citation label"
    assert any("water_cycle" in c for c in answer.citations)


def test_build_prompt_includes_context_and_question():
    chunks = [(Chunk("a.md", 0, "alpha text"), 0.9), (Chunk("b.md", 1, "beta text"), 0.5)]
    prompt = build_prompt("what is alpha?", chunks)
    assert "QUESTION: what is alpha?" in prompt
    assert "[a.md#0] alpha text" in prompt
    assert "[b.md#1] beta text" in prompt


def test_answer_without_documents_is_graceful():
    pipeline = RagPipeline()
    answer = pipeline.answer("anything?", top_k=3)  # nothing ingested
    assert "could not find" in answer.text.lower()
    assert answer.citations == []

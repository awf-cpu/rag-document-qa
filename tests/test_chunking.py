"""Tests for document chunking."""

import pytest

from rag_qa.loader import chunk_text


def test_chunk_text_basic_windowing():
    text = " ".join(str(i) for i in range(100))  # 100 words
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    # step = 20, so windows start at 0,20,40,60,80 -> 5 chunks
    assert len(chunks) == 5
    assert chunks[0].split()[:3] == ["0", "1", "2"]


def test_chunk_text_overlap_shared_words():
    text = " ".join(str(i) for i in range(50))
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    first_tail = chunks[0].split()[-5:]
    second_head = chunks[1].split()[:5]
    assert first_tail == second_head  # overlap preserves boundary context


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_short_text_single_chunk():
    assert chunk_text("just a few words", chunk_size=60) == ["just a few words"]


def test_chunk_text_invalid_params():
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=10, overlap=10)

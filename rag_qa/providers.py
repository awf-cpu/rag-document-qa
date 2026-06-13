"""Provider abstractions for embeddings and language models.

Two abstract base classes define the contract used by the rest of the package.
Each has a deterministic, offline ``Mock`` implementation that needs no network
access or API keys, plus an optional OpenAI implementation that activates only
when ``OPENAI_API_KEY`` is present and the ``openai`` package is installed.
"""

from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod
from typing import List, Sequence


# --------------------------------------------------------------------------- #
# Embedding providers
# --------------------------------------------------------------------------- #
class EmbeddingProvider(ABC):
    """Turns text into fixed-length numeric vectors."""

    #: Dimensionality of the vectors produced by this provider.
    dim: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of texts, returning one vector per input text."""
        raise NotImplementedError


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based embeddings.

    The embedding for a piece of text is built from a bag-of-words style hash:
    every token is hashed into one of ``dim`` buckets and contributes a weight.
    The resulting vectors are L2-normalised, so cosine similarity is meaningful
    and texts sharing vocabulary land close together — enough to demonstrate and
    test a full RAG pipeline without any external service.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for token in _tokenize(text):
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign
        return _l2_normalize(vec)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embeddings backed by the OpenAI API (optional, needs OPENAI_API_KEY)."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI  # imported lazily so it stays optional

        self._client = OpenAI()
        self.model = model
        self.dim = 1536  # text-embedding-3-small default dimensionality

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in resp.data]


def get_embedding_provider(name: str | None = None) -> EmbeddingProvider:
    """Return an embedding provider.

    ``name`` may be ``"mock"`` or ``"openai"``. When omitted it is read from the
    ``RAG_EMBED_PROVIDER`` env var, defaulting to ``"mock"`` so the package works
    offline out of the box.
    """
    name = (name or os.getenv("RAG_EMBED_PROVIDER") or "mock").lower()
    if name == "mock":
        return MockEmbeddingProvider()
    if name == "openai":
        return OpenAIEmbeddingProvider()
    raise ValueError(f"unknown embedding provider: {name!r}")


# --------------------------------------------------------------------------- #
# LLM providers
# --------------------------------------------------------------------------- #
class LLMProvider(ABC):
    """Generates an answer string given a fully built prompt."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return the model's completion for ``prompt``."""
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    """Deterministic offline "LLM".

    It does not invent text. Instead it parses the grounded context blocks out of
    the prompt and returns a templated, citation-bearing answer composed from the
    retrieved snippets. This keeps tests deterministic while still exercising the
    real prompt-construction and citation code paths.
    """

    def complete(self, prompt: str) -> str:
        question = _extract_section(prompt, "QUESTION:")
        context = _extract_section(prompt, "CONTEXT:")
        snippets = _parse_context_blocks(context)
        if not snippets:
            return "I could not find anything relevant in the provided documents."

        lines = [f"Based on the retrieved documents, regarding: {question.strip()}", ""]
        for source, text in snippets:
            first_sentence = text.strip().split(". ")[0].strip().rstrip(".")
            lines.append(f"- {first_sentence}. [{source}]")
        return "\n".join(lines)


class OpenAILLMProvider(LLMProvider):
    """Chat-completion backed answers (optional, needs OPENAI_API_KEY)."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI  # lazy import keeps it optional

        self._client = OpenAI()
        self.model = model

    def complete(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer strictly from the provided context and cite "
                        "every claim with the [source] tag it came from."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""


def get_llm_provider(name: str | None = None) -> LLMProvider:
    """Return an LLM provider (``"mock"`` or ``"openai"``).

    Reads ``RAG_LLM_PROVIDER`` when ``name`` is omitted, defaulting to ``"mock"``.
    """
    name = (name or os.getenv("RAG_LLM_PROVIDER") or "mock").lower()
    if name == "mock":
        return MockLLMProvider()
    if name == "openai":
        return OpenAILLMProvider()
    raise ValueError(f"unknown LLM provider: {name!r}")


# --------------------------------------------------------------------------- #
# Small text helpers (kept here to avoid extra dependencies)
# --------------------------------------------------------------------------- #
def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenisation."""
    token, out = [], []
    for ch in text.lower():
        if ch.isalnum():
            token.append(ch)
        elif token:
            out.append("".join(token))
            token = []
    if token:
        out.append("".join(token))
    return out


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def _extract_section(prompt: str, header: str) -> str:
    """Return the text following ``header`` up to the next ALL-CAPS header."""
    idx = prompt.find(header)
    if idx == -1:
        return ""
    rest = prompt[idx + len(header):]
    # stop at the next "WORD:" style header on its own logic
    for marker in ("QUESTION:", "CONTEXT:", "INSTRUCTIONS:", "ANSWER:"):
        if marker == header:
            continue
        pos = rest.find(marker)
        if pos != -1:
            rest = rest[:pos]
    return rest.strip()


def _parse_context_blocks(context: str) -> List[tuple[str, str]]:
    """Parse ``[source] text`` blocks produced by the pipeline prompt builder."""
    blocks: List[tuple[str, str]] = []
    for raw in context.split("\n"):
        line = raw.strip()
        if line.startswith("[") and "]" in line:
            close = line.index("]")
            source = line[1:close]
            text = line[close + 1:].strip()
            if text:
                blocks.append((source, text))
    return blocks

"""Command-line entry point: ``python -m rag_qa.ask "question" --docs ./sample_docs``."""

from __future__ import annotations

import argparse
import sys

from .pipeline import RagPipeline
from .providers import get_embedding_provider, get_llm_provider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag_qa.ask",
        description="Ask a question grounded in a folder of documents (RAG).",
    )
    parser.add_argument("question", help="the question to answer")
    parser.add_argument(
        "--docs",
        default="./sample_docs",
        help="directory of .txt/.md/.pdf documents (default: ./sample_docs)",
    )
    parser.add_argument("--top-k", type=int, default=3, help="chunks to retrieve")
    parser.add_argument(
        "--embed-provider",
        default=None,
        help="embedding provider: mock|openai (default: env or mock)",
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        help="LLM provider: mock|openai (default: env or mock)",
    )
    args = parser.parse_args(argv)

    pipeline = RagPipeline(
        embedder=get_embedding_provider(args.embed_provider),
        llm=get_llm_provider(args.llm_provider),
    )
    n_chunks = pipeline.ingest(args.docs)
    if n_chunks == 0:
        print(f"No documents found in {args.docs!r}.", file=sys.stderr)
        return 1

    result = pipeline.answer(args.question, top_k=args.top_k)
    print(result.text)
    print("\nSources:")
    for chunk, score in result.sources:
        print(f"  [{chunk.source}#{chunk.chunk_id}] (score={score:.3f})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

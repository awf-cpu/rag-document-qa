# rag-document-qa

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![CI](https://github.com/awf-cpu/rag-document-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/awf-cpu/rag-document-qa/actions/workflows/ci.yml)

A small, dependency-light **Retrieval-Augmented Generation (RAG)** engine that answers questions over a folder of documents and **cites the source chunks** it used. It runs fully **offline** with deterministic mock providers, so you can read, run and test it without API keys, and switch to real providers (OpenAI) with an environment variable.

## The business problem

Teams sit on contracts, policies, manuals and case files that nobody can search by meaning, and a plain chatbot invents answers. RAG grounds the model in *your* documents and makes it show where each statement came from, so a person can verify it in seconds. That matters most where a wrong answer has a cost: legal, compliance, support, internal knowledge bases.

## Try it in one line

```bash
pip install -r requirements.txt && python -m rag_qa.ask "How does the water cycle work?" --docs ./sample_docs
```

Output (offline mock provider; scores vary slightly with the mock embedder):

```
Based on the retrieved documents, regarding: How does the water cycle work?

- The Water Cycle The water cycle, also known as the hydrologic cycle, describes the continuous movement of water on, above, and below the surface of the Earth. [water_cycle.txt#0]
- precipitation reaches the ground, some of it flows over the surface as runoff into rivers and oceans, while some soaks into the soil as infiltration, replenishing groundwater. [water_cycle.txt#3]

Sources:
  [water_cycle.txt#0] (score=0.543)
  [water_cycle.txt#3] (score=0.363)
  [water_cycle.txt#1] (score=0.333)
```

Real provider (needs `pip install openai`):

```bash
export OPENAI_API_KEY=...
python -m rag_qa.ask "What HTTP status codes mean success?" \
    --docs ./sample_docs --embed-provider openai --llm-provider openai
```

## Architecture

1. **Load and chunk** `.txt` / `.md` (and `.pdf` if `pypdf` is installed) into overlapping word windows (`loader.py`).
2. **Embed** each chunk through an `EmbeddingProvider` and store it in an in-memory NumPy cosine-similarity `VectorIndex` (`index.py`).
3. **Retrieve** the top-k chunks, then **build a grounded prompt** with `[source#id]` citation tags (`pipeline.py`).
4. **Answer** through an `LLMProvider` instructed to cite every claim (`providers.py`). Both providers have offline mock implementations, which keeps tests deterministic while exercising the real code paths.

FAISS, Chroma or pgvector can slot in behind the same `VectorIndex` interface.

## Tests

```bash
pip install pytest && pytest -q
```

12 tests: chunking (windowing, overlap, edge cases), embedding determinism, retrieval relevance for a known query, and an end-to-end check that the final answer carries a citation. CI runs them on Python 3.11 and 3.12 with no API keys.

## How this becomes a solution for your company

This repository is the teaching-size version of a pattern I build for clients: RAG over internal documents with source citations, access control, local or in-country hosting when data cannot leave the company, and guardrails for regulated domains. If you want an assistant grounded in your own documents, see **Work with me** below.

## Work with me

I am André Feix, founder of **Pindorama Tecnologia** (Florianópolis, Brazil). We build custom AI systems for companies and law firms: WhatsApp agents with compliance guardrails (LGPD, bar-association rules), RAG over documents, document and deadline automation, LLM routing with cost control, and local AI with data kept in Brazil.

- LinkedIn: https://www.linkedin.com/in/andrefeix
- Email: andrefeix777@gmail.com
- More: https://github.com/awf-cpu

## License

MIT © 2026 awf-cpu

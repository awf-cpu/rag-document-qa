# rag-document-qa

A small, dependency-light **Retrieval-Augmented Generation (RAG)** engine that
answers questions over a folder of documents and **cites the source chunks** it
used. It runs fully **offline** with deterministic mock providers, so you can
read, run, and test it without any API keys — and swap in real providers
(OpenAI) by setting an environment variable.

## Why it matters

RAG is the standard pattern for grounding an LLM in your own data and reducing
hallucination. The interesting engineering is rarely the model call — it's the
plumbing: clean provider abstractions, sensible chunking, a retrieval index, and
prompts that force the model to cite its sources. This repo demonstrates all of
that in a few hundred readable lines, with tests proving each stage works.

## Architecture (in 4 lines)

1. **Load & chunk** `.txt` / `.md` (and `.pdf` if `pypdf` is installed) into overlapping word windows.
2. **Embed** each chunk via an `EmbeddingProvider` and store it in an in-memory NumPy cosine-similarity `VectorIndex`.
3. **Retrieve** the top-k chunks for a question, then **build a grounded prompt** with `[source#id]` citation tags.
4. **Answer** through an `LLMProvider` instructed to cite every claim; both providers have offline mock implementations.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional real providers / pdf support:
pip install openai pypdf
```

## Usage

```bash
python -m rag_qa.ask "How does the water cycle work?" --docs ./sample_docs
```

Output (offline mock provider):

```
Based on the retrieved documents, regarding: How does the water cycle work?

- The water cycle, also known as the hydrologic cycle, describes the continuous movement of water on, above, and below the surface of the Earth [water_cycle.txt#0]
- Evaporation is the process by which the sun heats water in oceans, lakes, and rivers, turning it into water vapour that rises into the atmosphere [water_cycle.txt#1]

Sources:
  [water_cycle.txt#0] (score=0.612)
  [water_cycle.txt#1] (score=0.498)
  [photosynthesis.md#1] (score=0.121)
```

Use a real provider:

```bash
export OPENAI_API_KEY=sk-...
python -m rag_qa.ask "What HTTP status codes mean success?" \
    --docs ./sample_docs --embed-provider openai --llm-provider openai
```

## How it works

- **`providers.py`** — `EmbeddingProvider` / `LLMProvider` ABCs. The mock embedder
  produces L2-normalised, hash-bucketed bag-of-words vectors (deterministic, so
  texts sharing vocabulary are close). The mock LLM parses the grounded context
  out of the prompt and returns a templated, citation-bearing answer — it never
  invents facts, which keeps tests deterministic while exercising real code paths.
- **`loader.py`** — directory loading + overlapping word-window chunking.
- **`index.py`** — `VectorIndex` doing cosine similarity over a NumPy matrix
  (FAISS/Chroma would slot in behind the same interface).
- **`pipeline.py`** — ties it together and builds the grounded prompt.

## Tests

```bash
pytest -q
```

Covers chunking (windowing, overlap, edge cases), embedding determinism,
retrieval relevance for a known query, and an end-to-end check that the final
answer contains a citation.

## License

MIT © 2026 awf-cpu

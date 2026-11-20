# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local RAG assistant answering French questions about the CNIL guide *"Sécurité des données personnelles"* (2024, `pdfs/`). Answers must be grounded in the guide, cite fiche + page, and refuse when the corpus lacks the answer. Code comments, prompts and answers are in French where they touch the domain.

## Commands

The stack runs in Docker; the LLM runs **on the host** via Ollama (`ollama pull llama3.1:8b`; on Linux Ollama needs `OLLAMA_HOST=0.0.0.0` so containers reach it through `host.docker.internal`). Copy `.env.example` to `.env` first.

```bash
./demo.sh                          # build + start everything, ingest the PDF if the collection is empty
make start | stop | logs | status
make clean                         # docker compose down -v — wipes Qdrant data and HF model cache
make ingest                        # POST the CNIL PDF to the ingestion service (:8001)
make query Q="..." P=v2            # ask the generation service (:8002); P = v0|v1|v2|v2-faiss|v3 (default v0)
make test                          # ./test_queries.sh — sample queries against the running stack (not unit tests)
EVAL_PIPELINE=v2 make eval         # RAGAS benchmark (pip install -r evaluation/requirements.txt first)
```

Unit tests (pytest, run from repo root, no services needed; `test_chunking.py` parses the real PDF):

```bash
python -m pytest tests/
python -m pytest tests/test_rag_core.py::test_rrf_rewards_documents_found_by_both_retrievers
```

Frontend (Streamlit) is at http://localhost:7860.

## Architecture

Four compose services on `rag_network`: `qdrant`, `ingestion` (FastAPI :8001), `generation` (FastAPI :8002), `frontend` (Streamlit :7860).

- **`ingestion_service/`** is self-contained (does **not** use `rag_core`, reads its own env vars). `chunking.py` does two passes: split the PDF on the guide's `FICHE N - TITLE` headings (plus `AVANT-PROPOS` and the annex), then split each section into ~900-char chunks (`CHUNK_SIZE`/`CHUNK_OVERLAP`) so each fits E5's 512-token window. Every chunk's text starts with its section title (`article_id`); tests enforce this. Point IDs are `uuid5(source_document:chunk_index)`, so re-ingesting a document replaces its chunks idempotently. Startup fails if the existing collection's vector size doesn't match the embedding model.
- **`rag_core/`** is the shared library used by the generation service (copied into its image — that's why `generation` builds with the repo root as context). Config comes only from env vars via `rag_core/config.py:Settings`.
  - `store.py:ChunkStore` — dense search in Qdrant, plus in-memory BM25 and FAISS indexes **built from the vectors already stored in Qdrant** (no re-embedding). They rebuild lazily when the Qdrant point count changes. Also provides `neighbours()` for small-to-big context expansion.
  - `embeddings.py` — E5 requires `passage:` prefix at index time and `query:` at query time; keep these asymmetric prefixes consistent with ingestion.
  - `retrieval.py` — reciprocal rank fusion and cross-encoder reranking.
  - `pipelines.py:build_pipelines` — registry of named pipelines (`PIPELINES`). All share the same prompt (`prompts.py`) and LLM settings so benchmark differences come only from retrieval: `v0` dense+rerank with raw HTTP to Ollama; `v1` same as LCEL chain; `v2` BM25+dense RRF+rerank; `v2-faiss` same with FAISS; `v3` LangGraph agent (`agent.py`: plan → retrieve → grade → rewrite loop → generate → grounding check) using the same model in Ollama JSON mode for decisions.
  - `text.py` — refusal contract: the model emits `[HORS_CORPUS]` + `REFUSAL_MESSAGE`; `parse_answer` returns `(answer, refused)` and treats a tag after a partial answer as not refused. BM25 tokenization (`french_tokenize`) strips accents and French stopwords.
- **Pipeline output contract**: each pipeline returns a dict with `answer`, `raw`, `refused`, `docs`, `latency_ms`, and optionally `llm_calls`/`trace` (v3). `generation_service/app.py` turns this into the `/query` response; an empty `raw` is a 502, not a refusal (to avoid inflating refusal scores). `/query` is a sync `def` deliberately so slow LLM calls don't block `/ping` health checks.
- **`evaluation/evaluation.py`** runs outside Docker against `http://localhost:8002/query`, loads `../.env`, and is configured by env vars: `EVAL_PIPELINE`, `RUN_NAME`, `EVAL_MODE` (`collect`|`evaluate`|`full`|`retry`), `EVAL_TOP_K`, `JUDGE_PROVIDER` (ollama/groq/mistral/anthropic) + `JUDGE_MODEL`. It saves after every question and resumes; results go to `evaluation/results/` named by pipeline. 40 questions across 5 categories (paraphrase, two-fiche, everyday wording, out-of-scope, not-covered); refusal correctness is scored separately and refused answers are excluded from RAGAS.

## Notes

- Everything runs on CPU (GPU unsupported on the dev machine); collecting answers for a full benchmark takes 1–2 h.
- Set `GEN_TEMPERATURE=0` for benchmark comparisons; with 10 questions per category, score differences under ~0.1 are noise.
- `OLLAMA_NUM_CTX` defaults to 8192 so the v3 agent's ~7,500-char context isn't silently truncated.

## Rules
- Tests must never download models or call the network: use fakes for the LLM, embeddings, reranker, and QdrantClient(":memory:").
- Run `pytest -q` and `ruff check .` before saying a task is done.
- Never change pipeline behaviour while writing tests; if a test reveals a bug, stop and explain it first.
- Never commit `.env`. Never push.
- No em dashes in documentation.
- Ask before installing, upgrading or removing any package.
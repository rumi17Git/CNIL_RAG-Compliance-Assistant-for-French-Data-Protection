# CNIL RAG – Compliance Assistant for French Data Protection

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-✓-2496ED)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production‑ready **Retrieval‑Augmented Generation (RAG)** system that answers questions about the CNIL guide on personal data security. It combines dense retrieval (multilingual‑E5), cross‑encoder reranking, and a local LLaMA 3.1 8B model served by Ollama, so documents never leave the machine. Answers are grounded in the guide and cite the fiche and page they come from.

---

## Table of Contents

- [Why this project?](#why-this-project)
- [Architecture](#architecture)
- [Features](#features)
- [Results](#results)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Evaluation](#evaluation)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Why this project?

French organisations must comply with the GDPR and the CNIL's recommendations on personal data security. Off-the-shelf LLMs hallucinate on specific regulatory questions — they may cite incorrect article numbers, invent obligations, or produce plausible-sounding but legally wrong answers.

This project demonstrates how to build a **domain‑specific RAG assistant** that:

- Ingests the CNIL guide *"Sécurité des données personnelles"* (2024 edition).
- Splits it into semantically meaningful chunks respecting the document's legal structure.
- Stores vectors in Qdrant with rich metadata (article ID, page number, source document).
- Answers user questions in French, **strictly from the provided context**.
- Returns **explicit citations** so every claim is traceable to a source page and article.
- Refuses to answer when the corpus does not contain the relevant information.

It was built to demonstrate production ML engineering skills for internship applications.

---

## Architecture

```text
[PDF] ──► [Ingestion API: POST /ingest] ──► [Parse → Chunk → Embed (E5)]
                                                         │
                                                         ▼
                                              [Vector DB: Qdrant]
                                                         ▲
                                                         │ (top-k chunks + metadata)
[User Query] ──► [Generation API: POST /query] ──────────┘
                         │
                         ▼
              [Cross-Encoder Reranker]
                         │
                         ▼
              [Ollama / LLaMA 3.1 8B, local] ──► [Grounded Response + Citations]
```

Three decoupled microservices communicating over a Docker bridge network (`rag_network`).

| Service | Responsibility |
|---|---|
| **Ingestion service** (FastAPI) | Parses PDF, cleans text, splits by fiche then by size, embeds with `multilingual-e5-base`, upserts to Qdrant with metadata |
| **Generation service** (FastAPI) | Embeds query, retrieves top-k chunks, reranks with cross-encoder, builds strict grounding prompt, calls the local model through Ollama |
| **Qdrant** | Vector database — similarity search with cosine distance, persistent storage volume |

---

## Features

**ML & Retrieval**
- **Asymmetric embedding** — `passage:` prefix for document chunks at index time, `query:` prefix for user questions at retrieval time (E5 requirement)
- **Two-pass chunking** — first splits the guide into its 25 fiches (plus foreword and annex), then splits each fiche into chunks of about 900 characters with overlap, so every chunk fits in the embedding model's 512-token window. Each chunk starts with its fiche title.
- **Cross-encoder reranking** — reorders retrieved chunks by relevance before generation
- **Strict grounding prompt** — when the context does not hold the answer, the model replies with a `[HORS_CORPUS]` tag and *"Je ne trouve pas cette information dans les documents fournis"*; the API returns `refused: true`

**Production Engineering**
- **Fully containerised** — `docker-compose up` brings the entire stack up in one command
- **Idempotent ingestion** — `POST /ingest` replaces a document's old chunks, so re-ingesting never creates duplicates, and reports how many chunks exceed the token limit
- **Health checks** — services start in order once their dependencies are ready
- **Metadata citations** — every answer includes `source_document`, `page_number`, and `article_id` for full traceability
- **Persistent volumes** — Qdrant data and HuggingFace model cache survive container restarts
- **Makefile + demo script** — one-command startup with automatic ingestion if the collection is empty
- **RAGAS evaluation suite** — 40 questions in 5 categories, scored with RAGAS and a configurable judge (local Ollama, Groq, Mistral or Anthropic), plus refusal and false-refusal rates; scoring is resumable and saves after every question

---

## Pipelines

All pipelines share the same prompt and the same local model. They differ only in how they find context, so their scores can be compared directly. Choose one with the `pipeline` field of `POST /query`, `make query P=v2`, or the menu in the Streamlit app.

| Pipeline | Retrieval |
|---|---|
| `v0` | Dense search in Qdrant (E5 embeddings), then cross-encoder reranking. The original code. |
| `v1` | The same method rebuilt as a LangChain (LCEL) chain. |
| `v2` | Hybrid search: BM25 keyword search and dense search, merged with reciprocal rank fusion, then reranking. |
| `v2-faiss` | The same hybrid search with an in-memory FAISS index instead of Qdrant. |
| `v3` | A LangGraph agent around the `v2` retriever (see below). |

### The v3 agent

`v3` wraps the hybrid retriever in a small LangGraph graph aimed at the two weak spots the benchmark found: everyday wording and questions spanning two fiches.

```text
plan -> retrieve -> grade --yes--> generate -> check --yes--> answer
          ^           | no                       | no
          |           v                          v
          +------- rewrite              regenerate once, stricter
        (at most 2 rewrites, then it answers anyway)
```

- **plan** splits a question about two topics into two searches.
- **retrieve** runs hybrid search per query and keeps the hits, then adds their neighbouring chunks within a 7,500-character budget ("small-to-big": precise search on small chunks, more context for the answer).
- **grade** asks whether the passages answer the question; if not, **rewrite** reformulates it in the guide's vocabulary ("Dropbox" becomes "cloud, sous-traitant").
- **check** asks whether every claim is supported by the context, and triggers one stricter regeneration if not.

The decisions use the same local model in Ollama's JSON mode. Every answer returns its `trace` (the steps taken) and `llm_calls`, shown in the Streamlit app under "Raisonnement de l'agent".

The shared logic lives in `rag_core/`. BM25 and FAISS are built from the chunks and vectors already stored in Qdrant, so nothing is embedded twice, and they rebuild automatically after a new ingestion.

---

## Results

All pipelines were run on the same **40 French questions in 5 categories**: paraphrases of the guide (10), questions combining two fiches (10), everyday wording (10), out-of-scope questions (3), and real CNIL topics the guide does not cover (7). Answers come from the local LLaMA 3.1 8B (6 chunks per question). They are scored with [RAGAS](https://github.com/explodinggradients/ragas), using `openai/gpt-oss-120b` on Groq as the judge, so the model that grades is not the model that answers.

| Pipeline | Correct refusals | False refusals | Context precision | Faithfulness | Answer relevancy |
|---|---|---|---|---|---|
| Page chunks (first version) | 10/10 | 3/30 | 0.90 | 0.87 | 0.89 |
| `v0` fiche chunks, dense search | 10/10 | 3/30 | 0.79 | 0.83 | 0.79 |
| `v1` same method in LangChain | 10/10 | 3/30 | 0.78 | 0.85 | 0.85 |
| `v2` hybrid BM25 + vectors | 10/10 | **1/30** | 0.80 | 0.81 | 0.83 |
| `v2-faiss` hybrid, FAISS index | 10/10 | **1/30** | 0.80 | 0.82 | 0.83 |

RAGAS scores are averaged over the answered questions (27 to 29 per run). Refused questions are counted in the refusal columns only.

**What the numbers show**

- **Refusals are reliable.** Every version refused all 10 questions the guide cannot answer, so the assistant does not invent answers.
- **Hybrid search reduces false refusals from 10% to 3%.** Both hybrid versions answered 9 of 10 everyday-wording questions, against 7 of 10 for dense search alone. Keyword search finds the right fiche when a question uses words the embeddings miss (for example "Google Drive" for a question about the cloud).
- **Most other differences are noise.** `v0` and `v1` use identical retrieval, yet their scores differ by up to 0.11 within a category, only because the model samples at temperature 0.2 and each category has 10 questions. Differences smaller than about 0.1 between pipelines should not be read as improvements.
- **Larger chunks give the model more context.** The first version (full pages, about 7,400 characters per question) scores higher on RAGAS than the fiche-based chunks (about 4,800 characters). The fiche chunks fix real problems (deleted headings, empty citations, text beyond the embedding model's 512-token window), so the next step keeps small chunks for search and passes the neighbouring text to the model.

Raw answers and per-question scores for every run are in `evaluation/results/`.

---

## Quick Start

**Prerequisites**
- Docker & Docker Compose (v2+)
- [Ollama](https://ollama.com) installed on your machine, with a model pulled: `ollama pull llama3.1:8b`
- On Linux, Ollama must accept connections from Docker: set `OLLAMA_HOST=0.0.0.0` for the Ollama service
- Python 3.10+ (only for running the evaluation suite locally)

### 1. Clone the repository

```bash
git clone https://github.com/rumi17Git/CNIL_RAG-Compliance-Assistant-for-French-Data-Protection.git
cd CNIL_RAG-Compliance-Assistant-for-French-Data-Protection
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Set OLLAMA_MODEL to the model you pulled
```

### 3. Start the system

```bash
./demo.sh
```

This will:
- Build the Docker images (first run: ~5–10 minutes for model downloads).
- Start Qdrant, ingestion service, and generation service.
- Automatically ingest the CNIL PDF if the collection is empty.

### 4. Ask a question

```bash
make query Q="Quelles sont les précautions pour les mots de passe ?"
```

Or directly via `curl`:

```bash
curl -X POST http://localhost:8002/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Que faire en cas de violation de données ?", "top_k": 3}'
```

### 5. Ingest a new document

```bash
curl -X POST http://localhost:8001/ingest -F "file=@./pdfs/your_document.pdf"
# Returns chunks_created, sections, max_tokens_per_chunk and chunks_over_token_limit
```

### 6. Run the full evaluation suite

```bash
pip install -r evaluation/requirements.txt
# In .env: JUDGE_PROVIDER=groq, JUDGE_MODEL=openai/gpt-oss-120b, GROQ_API_KEY=...
EVAL_PIPELINE=v2 make eval
# EVAL_MODE=collect only asks the questions, EVAL_MODE=evaluate only scores saved answers,
# EVAL_MODE=retry asks again only the questions that failed (timeouts).
# Results go to evaluation/results/ with the pipeline name in the file name.
# Asking the 40 questions takes 1 to 2 hours on a laptop CPU; scoring with Groq takes about 10 minutes.
```

---

## Usage Examples

**Question:**
> Quelles sont les précautions élémentaires pour les mots de passe ?

**Answer (truncated):**
```
Voici les précautions élémentaires pour les mots de passe selon la CNIL :

• Privilégier l'authentification multifacteur lorsque cela est possible,
  en particulier pour les accès depuis l'extérieur du réseau.
• Limiter le nombre de tentatives d'accès et bloquer le compte temporairement
  si la limite est atteinte.
• La CNIL met à disposition un outil pour calculer la complexité des mots de
  passe selon le cas d'usage (cnil.fr).
• Stocker les mots de passe de façon sécurisée transformés (« hash ») avec
  une fonction spécifiquement conçue à cette fin et utilisant toujours un sel.

[Source: cnil_guide_securite_personnelle_2024.pdf | Page 21 | FICHE 4 - AUTHENTIFIER LES UTILISATEURS]
```

---

## Project Structure

```text
.
├── docker-compose.yml
├── .env.example
├── Makefile
├── demo.sh
├── test_queries.sh
├── evaluation/
│   ├── evaluation.py
│   ├── requirements.txt
│   └── results/
├── frontend/
│   ├── dockerfile
│   ├── requirements.txt
│   └── app.py
├── rag_core/                 # shared retrieval, prompt and pipelines
├── tests/
├── ingestion_service/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── chunking.py
│   └── app.py
├── generation_service/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
└── pdfs/
    └── cnil_guide_securite_personnelle_2024.pdf
```

---

## Evaluation

The script `evaluation/evaluation.py` uses RAGAS to compute three core metrics:

- **Context Precision** — proportion of retrieved chunks that are directly relevant to the question.
- **Faithfulness** — fraction of answer statements that can be strictly inferred from the retrieved context (measures hallucination absence).
- **Answer Relevancy** — how directly the answer addresses the user's question (measured via embedding similarity between question and answer).

It also checks refusals: out-of-scope and not-covered questions should be refused, answerable ones should not. Refused answers are not scored by RAGAS. The evaluation scores one question at a time and saves after each one, so a run stopped by an API rate limit continues where it left off. Dependencies in `evaluation/requirements.txt` are separate from the Docker stack.

---

## Known Limitations

- **Broad query retrieval** — queries that span multiple FICHE sections (e.g., *"précautions élémentaires"* without specifying a domain) may retrieve topically adjacent chunks rather than the single most relevant article. A query expansion or HyDE (Hypothetical Document Embeddings) approach would improve precision further.
- **Single corpus** — the system is currently scoped to the CNIL security guide. `POST /ingest` accepts other PDFs, but the section detection and the prompt are tuned for this guide.
- **Synchronous ingestion** — `POST /ingest` waits until all chunks are embedded.
- **Small benchmark** — 10 questions per category means run-to-run variation of about 0.1 in RAGAS scores; only consistent differences are meaningful.
- **Slow answers on CPU** — the local 8B model takes about 2 minutes per answer with 6 chunks of context on a laptop CPU.
- **Hosted judge** — scoring uses a hosted model (Groq) for speed and reliability; only the answers and the public guide's passages are sent, the RAG itself stays local.
- **CPU-only embedding** — the Quadro P1000 GPU on the development machine is not supported by the current PyTorch build; embeddings run on CPU (~2–4s per batch). GPU acceleration would reduce ingestion time significantly.

---

## Future Improvements

- **Query expansion / HyDE** — generate a hypothetical answer to improve retrieval precision on broad questions.
- **Streaming responses** — improve perceived latency for long answers via FastAPI `StreamingResponse`.
- **Conversation memory** — add multi-turn chat history to the generation service.
- **Single-container HuggingFace Spaces deployment** — public live demo with ChromaDB replacing Qdrant for the in-memory demo environment.
- **Table and list preservation** — current chunking loses some structured content (numbered lists, comparison tables) from the CNIL guide.

---

## License

MIT © Rumman Jamil — feel free to use and adapt for your own portfolio.

---

## Acknowledgments

- [CNIL](https://www.cnil.fr) for the public guide (*Guide de la sécurité des données personnelles*, 2024).
- [Ollama](https://ollama.com) for local LLM inference.
- [Qdrant](https://qdrant.tech) for the vector database.
- [Hugging Face](https://huggingface.co) for the `multilingual-e5` models and cross-encoder.
- [RAGAS](https://github.com/explodinggradients/ragas) for the RAG evaluation framework.

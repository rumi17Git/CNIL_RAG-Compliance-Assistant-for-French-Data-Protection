# CNIL RAG – Compliance Assistant for French Data Protection

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-✓-2496ED)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production‑ready **Retrieval‑Augmented Generation (RAG)** system that answers questions about the CNIL guide on personal data security. It combines dense retrieval (multilingual‑E5), cross‑encoder reranking, and Groq's LLaMA 3.1 to deliver grounded, citation‑enriched answers in French.

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
              [Groq / LLaMA 3.1] ──► [Grounded French Response + Citations]
```

Three decoupled microservices communicating over a Docker bridge network (`rag_network`).

| Service | Responsibility |
|---|---|
| **Ingestion service** (FastAPI) | Parses PDF, cleans text, dual-pass chunking, embeds with `multilingual-e5-base`, upserts to Qdrant with metadata |
| **Generation service** (FastAPI) | Embeds query, retrieves top-k chunks, reranks with cross-encoder, builds strict French prompt, calls Groq LLaMA 3.1 |
| **Qdrant** | Vector database — similarity search with cosine distance, persistent storage volume |

---

## Features

**ML & Retrieval**
- **Asymmetric embedding** — `passage:` prefix for document chunks at index time, `query:` prefix for user questions at retrieval time (E5 requirement)
- **Dual-pass chunking** — structural regex splits first on legal boundaries (`FICHE N –`, `AVANT-PROPOS`, etc.), recursive character splitter as fallback for oversized sections
- **Cross-encoder reranking** — reorders retrieved chunks by relevance before generation (boosted Context Precision from ~0.84 → 1.00 on our test set)
- **Strict grounding prompt** — LLM returns *"Je ne trouve pas cette information dans les documents fournis"* rather than hallucinating when context is absent

**Production Engineering**
- **Fully containerised** — `docker-compose up` brings the entire stack up in one command
- **Async ingestion** — `POST /ingest` returns a job ID immediately (`202 Accepted`); `GET /ingest/status/{job_id}` polls progress — zero downtime for new documents
- **Metadata citations** — every answer includes `source_document`, `page_number`, and `article_id` for full traceability
- **Persistent volumes** — Qdrant data and HuggingFace model cache survive container restarts
- **Makefile + demo script** — one-command startup with automatic ingestion if the collection is empty
- **RAGAS evaluation suite** — offline scoring with Groq as judge LLM

---

## Results

Evaluated with [RAGAS](https://github.com/explodinggradients/ragas) using Groq LLaMA 3.1 as the judge LLM and `intfloat/multilingual-e5-base` for embedding similarity. Scores averaged over **10 French compliance questions** covering all major sections of the CNIL guide.

| Metric | Score |
|---|---|
| **Context Precision** | **1.00** |
| **Faithfulness** | **1.00** |
| **Answer Relevancy** | **0.92** |

**Interpretation**

- *Context Precision = 1.00* — Every retrieved chunk was directly relevant; no noisy or off-topic context was passed to the LLM.
- *Faithfulness = 1.00* — The model never invented information. Every statement in every answer is inferable from the retrieved CNIL text.
- *Answer Relevancy = 0.92* — Answers are highly on-topic. The 8% gap reflects cases where broad queries retrieve topically adjacent but not maximally precise chunks.

**Hallucination refusal example** — when asked *"Quel est le montant de l'amende en cas de non-conformité au RGPD ?"* (GDPR fine amount — not covered in the CNIL security guide), the system correctly responds:

> *Je ne trouve pas cette information dans les documents fournis.*

---

## Quick Start

**Prerequisites**
- Docker & Docker Compose (v2+)
- A Groq API key ([free tier](https://console.groq.com) works)
- Python 3.10+ (only for running the evaluation suite locally)

### 1. Clone the repository

```bash
git clone https://github.com/RummanJ17/cnil-rag.git
cd cnil-rag
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Open .env and add your GROQ_API_KEY
```

`.env.example`:

```ini
GROQ_API_KEY=your_key_here
EMBEDDING_MODEL=intfloat/multilingual-e5-base
QDRANT_HOST=qdrant
QDRANT_PORT=6333
COLLECTION_NAME=cnil_docs
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
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "/app/pdfs/your_document.pdf"}'
# Returns: {"job_id": "abc123", "status": "PROCESSING"}

curl http://localhost:8001/ingest/status/abc123
# Returns: {"job_id": "abc123", "status": "COMPLETED", "chunks_upserted": 114}
```

### 6. Run the full evaluation suite

```bash
cd evaluation
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
python evaluate.py
# Note: uses max_workers=1 to respect Groq free-tier rate limits (~25 min)
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
│   ├── evaluate.py
│   └── requirements.txt
├── ingestion_service/
│   ├── Dockerfile
│   ├── requirements.txt
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

The script `evaluation/evaluate.py` uses RAGAS to compute three core metrics:

- **Context Precision** — proportion of retrieved chunks that are directly relevant to the question.
- **Faithfulness** — fraction of answer statements that can be strictly inferred from the retrieved context (measures hallucination absence).
- **Answer Relevancy** — how directly the answer addresses the user's question (measured via embedding similarity between question and answer).

The evaluation uses `max_workers=1` to respect Groq free-tier rate limits (6,000 TPM). A full run completes in approximately 25 minutes. See `evaluation/requirements.txt` for dependencies — these are isolated from the main Docker stack and only needed for local evaluation.

---

## Known Limitations

- **Broad query retrieval** — queries that span multiple FICHE sections (e.g., *"précautions élémentaires"* without specifying a domain) may retrieve topically adjacent chunks rather than the single most relevant article. A query expansion or HyDE (Hypothetical Document Embeddings) approach would improve precision further.
- **Single corpus** — the system is currently scoped to the CNIL security guide. The async ingestion API (`POST /ingest`) supports additional documents without restart, but the retrieval prompt is tuned for CNIL regulatory French.
- **In-memory job tracker** — ingestion job status is stored in a Python dictionary and resets on container restart. Production replacement: Redis. A `# TODO` comment marks this in the code.
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
- [Groq](https://groq.com) for fast, free LLM inference.
- [Qdrant](https://qdrant.tech) for the vector database.
- [Hugging Face](https://huggingface.co) for the `multilingual-e5` models and cross-encoder.
- [RAGAS](https://github.com/explodinggradients/ragas) for the RAG evaluation framework.

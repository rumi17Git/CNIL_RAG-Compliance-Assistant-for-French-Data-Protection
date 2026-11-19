"""Generation service: answers a question with one of the RAG pipelines in rag_core."""
from contextlib import asynccontextmanager
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag_core.config import Settings
from rag_core.embeddings import E5Embeddings
from rag_core.llm import make_json_llm, make_llm
from rag_core.pipelines import PIPELINES, Components, build_pipelines
from rag_core.store import ChunkStore

settings = Settings()
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once at startup (CPU)
    embedder = SentenceTransformer(settings.embedding_model, device="cpu")
    reranker = CrossEncoder(settings.reranker_model, device="cpu")
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    store = ChunkStore(client, settings.collection_name, E5Embeddings(embedder))
    state["store"] = store
    state["pipelines"] = build_pipelines(Components(settings, store, reranker, make_llm(settings),
                                                    json_llm=make_json_llm(settings)))
    yield


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 6
    pipeline: Optional[str] = None  # v0, v1, v2, v2-faiss, v3; default from DEFAULT_PIPELINE


class Citation(BaseModel):
    text: str
    source_document: str
    page_number: int
    page_end: Optional[int] = None
    article_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    pipeline: str
    latency_ms: int
    llm_calls: int = 1
    trace: List[str] = []   # the agent's steps (v3); empty for the other pipelines
    citations: List[Citation]


@app.get("/ping")
def ping():
    return {"message": "Generation service is alive (using local Ollama)"}


@app.get("/pipelines")
def pipelines():
    return {"available": PIPELINES, "default": settings.default_pipeline}


# A plain "def" endpoint runs in a worker thread, so a slow answer (a local LLM can
# take a minute) doesn't block /ping and make the health check fail.
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    name = req.pipeline or settings.default_pipeline
    if name not in state["pipelines"]:
        raise HTTPException(status_code=400, detail=f"Unknown pipeline '{name}'. Choose from {PIPELINES}.")
    try:
        out = state["pipelines"][name](req.question, req.top_k)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Ollama request error: {e}")
    except Exception as e:  # e.g. Ollama unreachable through LangChain, Qdrant down
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")

    if not out["docs"]:
        raise HTTPException(status_code=404, detail="No relevant documents found. Run the ingestion first.")
    # An empty reply is a failure, not a refusal: counting it as a refusal would
    # inflate the refusal score in the evaluation.
    if not out["raw"].strip():
        raise HTTPException(status_code=502, detail="Ollama returned an empty answer")

    citations = [
        Citation(text=d.page_content,
                 source_document=d.metadata.get("source_document", "unknown"),
                 page_number=d.metadata.get("page_number", 0),
                 page_end=d.metadata.get("page_end"),
                 article_id=d.metadata.get("article_id"))
        for d in out["docs"]
    ]
    return QueryResponse(answer=out["answer"], refused=out["refused"], pipeline=name,
                         latency_ms=out["latency_ms"], llm_calls=out.get("llm_calls", 1),
                         trace=out.get("trace", []), citations=citations)

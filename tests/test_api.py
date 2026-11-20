"""Tests for the generation service API (generation_service/app.py).

The models the service loads at startup (embedding model, reranker, Ollama LLMs)
are replaced with fakes and Qdrant runs in memory, so nothing is downloaded and
no server needs to be running.
"""
import hashlib
import importlib.util
import math
import os
import sys
import types
from typing import Any, List, Optional

import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.llms import LLM
from pydantic import Field
from qdrant_client import QdrantClient
from qdrant_client.http import models

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

# The service imports sentence_transformers at module level. It is replaced by
# fakes below, so a stub is enough when it isn't installed (requirements-dev.txt
# leaves out torch).
try:
    import sentence_transformers  # noqa: F401
except ImportError:
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = stub.CrossEncoder = None
    sys.modules["sentence_transformers"] = stub

import rag_core.pipelines as pipelines  # noqa: E402
from rag_core.pipelines import PIPELINES  # noqa: E402
from rag_core.text import french_tokenize  # noqa: E402

# Loaded under its own name: ingestion_service/app.py is also called "app"
_spec = importlib.util.spec_from_file_location(
    "generation_app", os.path.join(ROOT, "generation_service", "app.py"))
gen_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_app)

QUESTION = "Quelles sont les précautions pour les mots de passe ?"
ANSWER = "Selon la FICHE 4, il faut privilégier l'authentification multifacteur."
# One JSON reply answers every agent decision: plan, grade, rewrite and check
JSON_REPLY = '{"requetes": [], "pertinent": "oui", "requete": "", "appuye": "oui"}'

DOC = "cnil_guide_securite_personnelle_2024.pdf"
SECTIONS = [
    ("FICHE 4 - AUTHENTIFIER LES UTILISATEURS", 4, 21, [
        "Privilégier l'authentification multifacteur pour les mots de passe.",
        "Stocker les mots de passe de façon sécurisée avec un hachage et un sel.",
        "Limiter le nombre de tentatives d'accès aux comptes utilisateurs.",
    ]),
    ("FICHE 15 - SAUVEGARDER", 15, 48, [
        "Effectuer des sauvegardes régulières des données.",
        "Stocker les sauvegardes dans un endroit sûr, hors site.",
    ]),
    ("FICHE 6 - TRACER LES OPÉRATIONS", 6, 26, [
        "Prévoir un système de journalisation des accès et des opérations.",
    ]),
]


def make_chunks():
    chunks = []
    for title, fiche, page, texts in SECTIONS:
        for text in texts:
            chunks.append({"source_document": DOC, "article_id": title, "fiche": fiche,
                           "page_number": page, "page_end": page, "chunk_index": len(chunks),
                           "text_content": f"{title}\n{text}"})
    return chunks


# ── Fakes ─────────────────────────────────────────────────────────────────────
DIM = 128


def hash_embed(text: str) -> List[float]:
    vec = [0.0] * DIM
    vec[0] = 0.01  # never a zero vector
    for token in french_tokenize(text):
        if token not in ("query", "passage"):  # the E5 prefixes
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


class FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.array(hash_embed(texts))
        return np.array([hash_embed(t) for t in texts])


class FakeCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return [len(set(french_tokenize(q)) & set(french_tokenize(d))) for q, d in pairs]


class FakeLLM(LLM):
    reply: str = ""
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs) -> str:
        self.calls.append(prompt)
        return self.reply


# ── Fixture ───────────────────────────────────────────────────────────────────
@pytest.fixture
def service(monkeypatch):
    """service(chunks=..., reply=...) -> (TestClient, answer_prompts).

    answer_prompts collects every prompt sent to the answering model, whether
    through Ollama's raw API (v0) or the LangChain LLM (v1 and later)."""
    clients = []

    def start(chunks=None, reply=ANSWER):
        chunks = make_chunks() if chunks is None else chunks
        qdrant = QdrantClient(":memory:")
        name = gen_app.settings.collection_name
        qdrant.create_collection(name, vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))
        if chunks:
            qdrant.upsert(name, points=[models.PointStruct(id=i, vector=hash_embed(c["text_content"]), payload=c)
                                        for i, c in enumerate(chunks)])

        llm = FakeLLM(reply=reply)
        json_llm = FakeLLM(reply=JSON_REPLY)

        def fake_generate_raw(settings, prompt):
            llm.calls.append(prompt)
            return reply

        monkeypatch.setattr(gen_app, "SentenceTransformer", FakeSentenceTransformer)
        monkeypatch.setattr(gen_app, "CrossEncoder", FakeCrossEncoder)
        monkeypatch.setattr(gen_app, "QdrantClient", lambda *a, **k: qdrant)
        monkeypatch.setattr(gen_app, "make_llm", lambda settings: llm)
        monkeypatch.setattr(gen_app, "make_json_llm", lambda settings: json_llm)
        monkeypatch.setattr(pipelines, "generate_raw", fake_generate_raw)  # v0 calls Ollama directly

        client = TestClient(gen_app.app)
        client.__enter__()  # runs the lifespan: loads the (fake) models
        clients.append(client)
        return client, llm.calls

    yield start
    for client in clients:
        client.__exit__(None, None, None)


def ask(client, pipeline=None, question=QUESTION, top_k=2):
    body = {"question": question, "top_k": top_k}
    if pipeline is not None:
        body["pipeline"] = pipeline
    return client.post("/query", json=body)


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_ping(service):
    client, _ = service()
    r = client.get("/ping")
    assert r.status_code == 200
    assert "alive" in r.json()["message"]


def test_pipelines(service):
    client, _ = service()
    r = client.get("/pipelines")
    assert r.status_code == 200
    assert r.json() == {"available": PIPELINES, "default": gen_app.settings.default_pipeline}


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_query_every_pipeline(service, pipeline):
    client, prompts = service()
    r = ask(client, pipeline)
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["pipeline"] == pipeline
    assert body["answer"] == ANSWER
    assert body["refused"] is False
    assert isinstance(body["latency_ms"], int) and body["latency_ms"] >= 0

    # Citations carry the chunk metadata, and the password fiche is found
    chunks = {c["text_content"]: c for c in make_chunks()}
    assert body["citations"]
    for cit in body["citations"]:
        chunk = chunks[cit["text"]]
        assert cit["source_document"] == DOC
        assert cit["page_number"] == chunk["page_number"]
        assert cit["page_end"] == chunk["page_end"]
        assert cit["article_id"] == chunk["article_id"]
    assert any(c["article_id"].startswith("FICHE 4") for c in body["citations"])

    # The answering model got the question and the cited passages
    assert prompts and QUESTION in prompts[-1]
    for cit in body["citations"]:
        assert cit["text"] in prompts[-1]

    if pipeline == "v3":
        assert body["llm_calls"] > 1   # plan, grade, generate, check...
        assert body["trace"] and body["trace"][0].startswith("plan:")
    else:
        assert body["llm_calls"] == 1
        assert body["trace"] == []
        assert len(body["citations"]) <= 2   # top_k (v3 adds neighbouring chunks)


def test_query_without_pipeline_uses_the_default(service):
    client, _ = service()
    r = ask(client)
    assert r.status_code == 200, r.text
    assert r.json()["pipeline"] == gen_app.settings.default_pipeline


def test_unknown_pipeline_returns_400(service):
    client, prompts = service()
    r = ask(client, "v99")
    assert r.status_code == 400
    assert "v99" in r.json()["detail"]
    for name in PIPELINES:
        assert name in r.json()["detail"]
    assert prompts == []   # the model is never called


@pytest.mark.parametrize("reply", ["", "   \n"], ids=["empty", "whitespace"])
@pytest.mark.parametrize("pipeline", PIPELINES)
def test_empty_model_reply_returns_502(service, pipeline, reply):
    client, _ = service(reply=reply)
    r = ask(client, pipeline)
    assert r.status_code == 502
    assert r.json()["detail"] == "Ollama returned an empty answer"


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_empty_collection_returns_404(service, pipeline):
    client, _ = service(chunks=[])
    r = ask(client, pipeline)
    assert r.status_code == 404
    assert "ingestion" in r.json()["detail"]

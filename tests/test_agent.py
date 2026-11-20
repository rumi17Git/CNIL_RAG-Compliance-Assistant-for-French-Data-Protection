"""Tests for the v3 LangGraph agent (rag_core/agent.py).

No Ollama, no model downloads: the agent runs on the real CNIL guide chunks
(ingestion_service/chunking.py) stored in an in-memory Qdrant, with a fake
embedding model, a fake reranker and a fake LLM that replies according to the
prompt it receives.
"""
import hashlib
import math
import os
import sys
from dataclasses import replace
from typing import Any, List, Optional

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from pydantic import Field
from qdrant_client import QdrantClient
from qdrant_client.http import models

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ingestion_service"))
import chunking  # noqa: E402

from rag_core.agent import build_agent, parse_json  # noqa: E402
from rag_core.config import Settings  # noqa: E402
from rag_core.pipelines import Components, _hybrid_retrieve  # noqa: E402
from rag_core.prompts import format_context  # noqa: E402
from rag_core.store import ChunkStore  # noqa: E402
from rag_core.text import REFUSAL_MESSAGE, REFUSAL_TAG, french_tokenize, parse_answer  # noqa: E402

PDF_NAME = "cnil_guide_securite_personnelle_2024.pdf"
PDF = os.path.join(ROOT, "pdfs", PDF_NAME)
COLLECTION = "test_chunks"

QUESTION = "Quelles sont les précautions élémentaires pour les mots de passe ?"
ANSWER = "Selon la FICHE 4, il faut privilégier l'authentification multifacteur."
STRICT_ANSWER = "Le guide recommande l'authentification multifacteur (FICHE 4)."
REFUSAL = f"{REFUSAL_TAG} {REFUSAL_MESSAGE}"

GRADE_YES = '{"pertinent": "oui"}'
GRADE_NO = '{"pertinent": "non"}'
CHECK_YES = '{"appuye": "oui"}'
CHECK_NO = '{"appuye": "non"}'


# ── Fakes ─────────────────────────────────────────────────────────────────────
class HashEmbeddings(Embeddings):
    """Bag of words hashed into a fixed-size vector: similar wording -> similar vectors."""
    dim = 256

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        vec[0] = 0.01  # never a zero vector (cosine would be undefined)
        for token in french_tokenize(text):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class OverlapReranker:
    """Scores a (question, passage) pair by the number of shared tokens."""

    def predict(self, pairs):
        return [len(set(french_tokenize(q)) & set(french_tokenize(d))) for q, d in pairs]


# Markers that identify each prompt. Retry and strict prompts embed the answer
# template, so they are tested before the plain answer prompt.
PROMPT_MARKERS = [
    ("plan", "Tu prépares une recherche"),
    ("grade", "Ces extraits contiennent-ils"),
    ("rewrite", "Reformule-la"),
    ("check", "Chaque affirmation de la réponse est-elle appuyée"),
    ("retry", "ont été jugés pertinents"),
    ("strict", "une première réponse contenait"),
    ("answer", "RÉPONSE ("),
]


def prompt_kind(prompt: str) -> str:
    for kind, marker in PROMPT_MARKERS:
        if marker in prompt:
            return kind
    raise AssertionError(f"Unrecognised prompt: {prompt[:200]!r}")


class ScriptedLLM(LLM):
    """Replies according to the kind of prompt. A reply can be a string or a list
    of strings consumed one call at a time (the last one repeats). A prompt kind
    with no scripted reply fails the test, so unexpected steps are caught."""
    script: dict = Field(default_factory=dict)
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs) -> str:
        kind = prompt_kind(prompt)
        self.calls.append((kind, prompt))
        if kind not in self.script:
            raise AssertionError(f"Unexpected '{kind}' call")
        reply = self.script[kind]
        if isinstance(reply, list):
            n = sum(1 for k, _ in self.calls if k == kind)
            return reply[min(n, len(reply)) - 1]
        return reply

    def kinds(self) -> List[str]:
        return [k for k, _ in self.calls]

    def prompts(self, kind: str) -> List[str]:
        return [p for k, p in self.calls if k == kind]


def make_llm(**overrides) -> ScriptedLLM:
    script = {"plan": '{"requetes": []}', "grade": GRADE_YES, "answer": ANSWER, "check": CHECK_YES}
    script.update(overrides)
    return ScriptedLLM(script=script)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def store():
    with open(PDF, "rb") as f:
        chunks = chunking.parse_pdf(f.read(), PDF_NAME)
    embeddings = HashEmbeddings()
    client = QdrantClient(":memory:")
    client.create_collection(COLLECTION, vectors_config=models.VectorParams(
        size=HashEmbeddings.dim, distance=models.Distance.COSINE))
    vectors = embeddings.embed_documents([c["text_content"] for c in chunks])
    client.upsert(COLLECTION, points=[models.PointStruct(id=i, vector=v, payload=c)
                                      for i, (c, v) in enumerate(zip(chunks, vectors))])
    return ChunkStore(client, COLLECTION, embeddings)


class RetrieveLog(list):
    """Each hybrid search the agent runs, as (query, docs)."""

    def queries(self) -> List[str]:
        return [q for q, _ in self]


@pytest.fixture
def agent(store):
    """agent(llm, **settings) -> (run, retrieve_log)"""
    def build(llm, **overrides):
        settings = replace(Settings(), agent_max_rewrites=2, agent_context_chars=7500,
                           agent_grounding_check=True, candidates_per_retriever=20)
        settings = replace(settings, **overrides)
        c = Components(settings, store, OverlapReranker(), llm, json_llm=llm)
        hybrid = _hybrid_retrieve(c, store.dense_qdrant)
        log = RetrieveLog()

        def recording_retrieve(question, top_k):
            docs = hybrid(question, top_k)
            log.append((question, docs))
            return docs

        return build_agent(c, recording_retrieve, llm), log
    return build


def steps(out) -> List[str]:
    return [t.split(":")[0].split(" ")[0] for t in out["trace"]]


# ── Paths through the graph ───────────────────────────────────────────────────
def test_normal_path(agent):
    llm = make_llm(plan=f'{{"requetes": ["{QUESTION}"]}}')
    run, log = agent(llm)
    out = run(QUESTION, 6)

    assert llm.kinds() == ["plan", "grade", "answer", "check"]
    assert steps(out) == ["plan", "retrieve", "grade", "generate", "check"]
    assert out["llm_calls"] == 4
    assert out["raw"] == ANSWER
    assert parse_answer(out["raw"]) == (ANSWER, False)
    assert log.queries() == [QUESTION]
    assert out["docs"]
    assert any(d.metadata["article_id"].startswith("FICHE 4") for d in out["docs"])
    # The answer and check prompts both carry the retrieved context
    context = format_context(out["docs"])
    assert context in llm.prompts("answer")[0]
    assert context in llm.prompts("check")[0]


def test_two_not_relevant_grades_lead_to_two_rewrites_then_an_answer(agent):
    llm = make_llm(grade=[GRADE_NO, GRADE_NO, GRADE_YES],
                   rewrite=['{"requete": "authentification"}', '{"requete": "mot de passe robuste"}'])
    run, log = agent(llm)
    out = run(QUESTION, 6)

    assert llm.kinds() == ["plan", "grade", "rewrite", "grade", "rewrite", "grade", "answer", "check"]
    assert out["raw"] == ANSWER
    assert out["llm_calls"] == 8
    # Each rewrite is searched alongside the original question
    assert log.queries() == [QUESTION, QUESTION, "authentification", QUESTION, "mot de passe robuste"]
    # The second rewrite is told which queries were already tried
    assert "authentification" in llm.prompts("rewrite")[1]


def test_rewrites_stop_at_the_limit_and_the_agent_answers_anyway(agent):
    llm = make_llm(grade=GRADE_NO, rewrite='{"requete": "authentification"}')
    run, _ = agent(llm, agent_max_rewrites=2)
    out = run(QUESTION, 6)

    assert llm.kinds() == ["plan", "grade", "rewrite", "grade", "rewrite", "grade", "answer", "check"]
    assert out["raw"] == ANSWER


def test_refusal_after_relevant_grade_triggers_exactly_one_retry(agent):
    llm = make_llm(answer=REFUSAL, retry=REFUSAL)
    run, _ = agent(llm)
    out = run(QUESTION, 6)

    assert llm.kinds() == ["plan", "grade", "answer", "retry"]
    assert steps(out) == ["plan", "retrieve", "grade", "generate", "retry"]
    assert parse_answer(out["raw"])[1] is True   # the refusal stands
    assert out["llm_calls"] == 4


def test_retry_that_answers_goes_on_to_the_grounding_check(agent):
    llm = make_llm(answer=REFUSAL, retry=ANSWER)
    run, _ = agent(llm)
    out = run(QUESTION, 6)

    assert llm.kinds() == ["plan", "grade", "answer", "retry", "check"]
    assert out["raw"] == ANSWER


def test_no_retry_when_grade_says_not_relevant(agent):
    llm = make_llm(grade=GRADE_NO, rewrite='{"requete": "authentification"}', answer=REFUSAL)
    run, _ = agent(llm)
    out = run(QUESTION, 6)

    assert "retry" not in llm.kinds()
    assert "check" not in llm.kinds()   # refusals are not grounding-checked
    assert llm.kinds()[-1] == "answer"
    assert parse_answer(out["raw"]) == (REFUSAL_MESSAGE, True)


def test_failed_grounding_check_triggers_one_stricter_regeneration(agent):
    llm = make_llm(check=CHECK_NO, strict=STRICT_ANSWER)
    run, _ = agent(llm)
    out = run(QUESTION, 6)

    # One regeneration, and its answer is not checked again
    assert llm.kinds() == ["plan", "grade", "answer", "check", "strict"]
    assert steps(out) == ["plan", "retrieve", "grade", "generate", "check", "regenerate"]
    assert out["raw"] == STRICT_ANSWER
    assert out["llm_calls"] == 5
    strict_prompt = llm.prompts("strict")[0]
    assert QUESTION in strict_prompt
    assert format_context(out["docs"]) in strict_prompt


def test_grounding_check_can_be_disabled(agent):
    llm = make_llm()
    run, _ = agent(llm, agent_grounding_check=False)
    run(QUESTION, 6)
    assert llm.kinds() == ["plan", "grade", "answer"]


# ── Plan ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("plan_reply, expected", [
    (f'{{"requetes": ["{QUESTION}"]}}', [QUESTION]),
    ('{"requetes": []}', [QUESTION]),
    ('{"requetes": ["mots de passe", "sauvegardes"]}', [QUESTION, "mots de passe", "sauvegardes"]),
    ('{"requetes": ["mots de passe", "sauvegardes", "journalisation"]}', [QUESTION, "mots de passe", "sauvegardes"]),
    (f'{{"requetes": ["  {QUESTION.upper()}  ", "sauvegardes"]}}', [QUESTION, "sauvegardes"]),
    ('{"requetes": ["", "   ", 42, null]}', [QUESTION]),
    ('{"requetes": "mots de passe"}', [QUESTION]),
    ('{"requetes": {"a": "mots de passe"}}', [QUESTION]),
], ids=["same", "empty", "split", "max-two-subqueries", "duplicate-of-question", "junk-items",
        "string-not-list", "object-not-list"])
def test_plan_always_keeps_the_original_question(agent, plan_reply, expected):
    llm = make_llm(plan=plan_reply)
    run, log = agent(llm)
    out = run(QUESTION, 6)

    assert log.queries() == expected
    assert log.queries()[0] == QUESTION
    assert out["trace"][0] == "plan: search " + " | ".join(repr(q) for q in expected)


# ── Invalid JSON ──────────────────────────────────────────────────────────────
GARBAGE = ["pas du JSON", "", "null", "[1, 2]", '{"requetes": ', "{'pertinent': 'non'}"]


@pytest.mark.parametrize("garbage", GARBAGE)
def test_invalid_json_falls_back_to_safe_defaults(agent, garbage):
    # plan -> search the question only; grade -> relevant; check -> supported
    llm = make_llm(plan=garbage, grade=garbage, check=garbage)
    run, log = agent(llm)
    out = run(QUESTION, 6)

    assert log.queries() == [QUESTION]
    assert llm.kinds() == ["plan", "grade", "answer", "check"]
    assert out["raw"] == ANSWER


@pytest.mark.parametrize("garbage", GARBAGE + ['{"requete": ""}'])
def test_invalid_rewrite_falls_back_to_the_question(agent, garbage):
    llm = make_llm(grade=[GRADE_NO, GRADE_YES], rewrite=garbage)
    run, log = agent(llm)
    out = run(QUESTION, 6)

    assert f"rewrite: {QUESTION!r}" in out["trace"]
    assert log.queries() == [QUESTION, QUESTION, QUESTION]
    assert out["raw"] == ANSWER


def test_json_wrapped_in_text_is_still_read(agent):
    llm = make_llm(grade=['Voici ma réponse : {"pertinent": "non"} .', GRADE_YES],
                   rewrite='```json\n{"requete": "authentification"}\n```')
    run, log = agent(llm)
    run(QUESTION, 6)
    assert log.queries() == [QUESTION, QUESTION, "authentification"]


@pytest.mark.parametrize("text, expected", [
    ('{"pertinent": "non"}', "non"),
    ('Réponse : {"pertinent": "non"}', "non"),
    ('{"autre": "non"}', "défaut"),
    ("pas du JSON", "défaut"),
    ("[1, 2]", "défaut"),
    ("null", "défaut"),
    ("", "défaut"),
    (None, "défaut"),
])
def test_parse_json(text, expected):
    assert parse_json(text, "pertinent", "défaut") == expected


# ── Context size ──────────────────────────────────────────────────────────────
def _hits(log: RetrieveLog) -> List[Document]:
    assert len(log) == 1
    return log[0][1]


def _total(docs: List[Document]) -> int:
    return sum(len(d.page_content) for d in docs)


@pytest.mark.parametrize("budget", [7500, 4000, 2500])
def test_context_stays_within_agent_context_chars(agent, budget):
    llm = make_llm()
    run, log = agent(llm, agent_context_chars=budget)
    out = run(QUESTION, 2)
    hits, docs = _hits(log), out["docs"]

    assert _total(hits) <= budget          # so the budget, not the hits, is the limit
    assert _total(docs) <= budget
    keys = [(d.metadata["source_document"], d.metadata["chunk_index"]) for d in docs]
    assert len(keys) == len(set(keys))     # no chunk twice
    assert keys == sorted(keys)            # document order
    for h in hits:                         # every hit is kept
        assert (h.metadata["source_document"], h.metadata["chunk_index"]) in keys
    for d in docs:                         # extra chunks are neighbours from the same section
        assert any(d.metadata["article_id"] == h.metadata["article_id"]
                   and abs(d.metadata["chunk_index"] - h.metadata["chunk_index"]) <= 1 for h in hits)
    assert format_context(docs) in llm.prompts("answer")[0]


def test_large_budget_adds_neighbouring_chunks(agent):
    run, log = agent(make_llm(), agent_context_chars=7500)
    out = run(QUESTION, 2)
    assert len(out["docs"]) > len(_hits(log))


def test_hits_are_kept_even_when_they_exceed_the_budget(agent):
    run, log = agent(make_llm(), agent_context_chars=100)
    out = run(QUESTION, 3)
    hits = _hits(log)
    assert _total(hits) > 100
    assert sorted(d.metadata["chunk_index"] for d in out["docs"]) == sorted(d.metadata["chunk_index"] for d in hits)

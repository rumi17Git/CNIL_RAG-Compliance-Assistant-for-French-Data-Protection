"""The pipelines being compared. They share the prompt and the answering model,
and differ only in how they retrieve context.

  v0        original code: dense search in Qdrant + reranker, plain Python
  v1        the same method rebuilt as a LangChain (LCEL) chain
  v2        hybrid search: BM25 + dense (Qdrant), fused with RRF, then reranker
  v2-faiss  the same hybrid search with a FAISS index instead of Qdrant
  v3        LangGraph agent around the v2 retriever: plans, grades the passages,
            rewrites the query, reads neighbouring chunks, checks its answer
"""
import time
from dataclasses import dataclass
from typing import Callable, Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .llm import generate_raw
from .prompts import PROMPT, format_context
from .retrieval import rerank, rrf_fuse
from .text import parse_answer

PIPELINES = ["v0", "v1", "v2", "v2-faiss", "v3"]


@dataclass
class Components:
    settings: object
    store: object       # ChunkStore
    reranker: object    # CrossEncoder (anything with .predict)
    llm: object         # LangChain LLM, e.g. OllamaLLM
    json_llm: object = None  # same model in JSON mode, for the agent's decisions (v3)


def _n_candidates(top_k: int) -> int:
    return max(top_k * 3, 15)


def _dense_retrieve(c: Components) -> Callable[[str, int], List[Document]]:
    def retrieve(question, top_k):
        return rerank(c.reranker, question, c.store.dense_qdrant(question, _n_candidates(top_k)), top_k)
    return retrieve


def _hybrid_retrieve(c: Components, dense: Callable) -> Callable[[str, int], List[Document]]:
    def retrieve(question, top_k):
        n = c.settings.candidates_per_retriever
        fused = rrf_fuse([dense(question, n), c.store.bm25(question, n)])
        return rerank(c.reranker, question, fused[:_n_candidates(top_k)], top_k)
    return retrieve


def _lcel_chain(retrieve: Callable, llm):
    """retrieve -> format context -> prompt -> LLM -> text"""
    return (
        RunnablePassthrough.assign(docs=RunnableLambda(lambda x: retrieve(x["question"], x["top_k"])))
        | RunnablePassthrough.assign(context=lambda x: format_context(x["docs"]))
        | RunnablePassthrough.assign(raw=PROMPT | llm | StrOutputParser())
    )


def build_pipelines(c: Components) -> Dict[str, Callable[[str, int], dict]]:
    def v0(question, top_k):
        docs = _dense_retrieve(c)(question, top_k)
        raw = generate_raw(c.settings, PROMPT.format(context=format_context(docs), question=question))
        return {"raw": raw, "docs": docs}

    chains = {
        "v1": _lcel_chain(_dense_retrieve(c), c.llm),
        "v2": _lcel_chain(_hybrid_retrieve(c, c.store.dense_qdrant), c.llm),
        "v2-faiss": _lcel_chain(_hybrid_retrieve(c, c.store.dense_faiss), c.llm),
    }

    def wrap(fn):
        def run(question: str, top_k: int) -> dict:
            start = time.perf_counter()
            out = fn(question, top_k)
            answer, refused = parse_answer(out["raw"])
            return {"answer": answer, "refused": refused, "docs": out["docs"], "raw": out["raw"],
                    "latency_ms": int((time.perf_counter() - start) * 1000),
                    "trace": out.get("trace", []), "llm_calls": out.get("llm_calls", 1)}
        return run

    runners = {"v0": wrap(v0)}
    for name, chain in chains.items():
        runners[name] = wrap(lambda q, k, chain=chain: chain.invoke({"question": q, "top_k": k}))
    if c.json_llm is not None:
        from .agent import build_agent
        runners["v3"] = wrap(build_agent(c, _hybrid_retrieve(c, c.store.dense_qdrant), c.json_llm))
    return runners

"""Fusion and reranking."""
from typing import List, Sequence

from langchain_core.documents import Document


def doc_key(doc: Document):
    m = doc.metadata
    return (m.get("source_document"), m.get("chunk_index"), doc.page_content[:40])


def rrf_fuse(ranked_lists: Sequence[List[Document]], k: int = 60) -> List[Document]:
    """Reciprocal rank fusion.

    Each document gets sum(1 / (k + rank)) over the lists it appears in. It only
    uses ranks, so BM25 scores and cosine similarities never need to be put on
    the same scale. k=60 is the value from the original RRF paper.
    """
    scores, docs = {}, {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs.setdefault(key, doc)
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


def rerank(reranker, question: str, docs: List[Document], top_k: int) -> List[Document]:
    """Order candidates with the cross-encoder and keep the best top_k."""
    if not docs:
        return []
    scores = reranker.predict([[question, d.page_content] for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    return [d for d, _ in ranked[:top_k]]

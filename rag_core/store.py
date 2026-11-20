"""Chunk store: dense search in Qdrant, plus BM25 and FAISS indexes built from the
same chunks.

BM25 and FAISS live in memory. They are built from the chunks (and their stored
vectors) already in Qdrant, so nothing is embedded twice, and they are rebuilt
automatically when the number of chunks changes (after a new ingestion).
"""
import threading
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from .text import french_tokenize


def _to_document(payload: dict) -> Document:
    meta = {k: v for k, v in payload.items() if k != "text_content"}
    return Document(page_content=payload.get("text_content", ""), metadata=meta)


class ChunkStore:
    def __init__(self, client: QdrantClient, collection_name: str, embeddings):
        self.client = client
        self.collection = collection_name
        self.embeddings = embeddings  # E5Embeddings
        self._lock = threading.Lock()
        self._count = -1
        self._bm25 = None
        self._faiss = None
        self._by_pos = {}  # (source_document, chunk_index) -> Document

    # ── Dense search in Qdrant ────────────────────────────────────────────────
    def dense_qdrant(self, question: str, k: int) -> List[Document]:
        vector = self.embeddings.embed_query(question)
        res = self.client.query_points(self.collection, query=vector, limit=k, with_payload=True)
        return [_to_document(p.payload) for p in res.points]

    # ── In-memory indexes ─────────────────────────────────────────────────────
    def _refresh(self):
        count = self.client.count(self.collection, exact=True).count
        if count == self._count:
            return
        with self._lock:
            if count == self._count:
                return
            points, offset = [], None
            while True:
                batch, offset = self.client.scroll(self.collection, limit=256, offset=offset,
                                                   with_payload=True, with_vectors=True)
                points.extend(batch)
                if offset is None:
                    break
            points.sort(key=lambda p: (p.payload.get("source_document", ""), p.payload.get("chunk_index", 0)))
            docs = [_to_document(p.payload) for p in points]
            self._by_pos = {(d.metadata.get("source_document"), d.metadata.get("chunk_index")): d for d in docs}
            if docs:
                self._bm25 = BM25Retriever.from_documents(docs, preprocess_func=french_tokenize)
                self._faiss = FAISS.from_embeddings(
                    text_embeddings=[(d.page_content, p.vector) for d, p in zip(docs, points)],
                    embedding=self.embeddings,
                    metadatas=[d.metadata for d in docs],
                    normalize_L2=True,  # L2 on unit vectors ranks the same as cosine, like Qdrant
                )
            else:
                self._bm25 = self._faiss = None
            self._count = count

    def bm25(self, question: str, k: int) -> List[Document]:
        self._refresh()
        if self._bm25 is None:
            return []
        return self._bm25_search(question, k)

    def _bm25_search(self, question: str, k: int) -> List[Document]:
        # BM25Retriever keeps k as an attribute; use its scorer directly so concurrent
        # requests with different k don't interfere.
        scores = self._bm25.vectorizer.get_scores(french_tokenize(question))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._bm25.docs[i] for i in ranked[:k] if scores[i] > 0]

    def neighbours(self, doc: Document, window: int = 1) -> List[Document]:
        """The chunk plus the chunks right before and after it in the same section,
        in document order ("small-to-big": search small chunks, read the bigger passage)."""
        self._refresh()
        src, idx = doc.metadata.get("source_document"), doc.metadata.get("chunk_index")
        section = doc.metadata.get("article_id")
        out = []
        for j in range(idx - window, idx + window + 1) if isinstance(idx, int) else []:
            d = self._by_pos.get((src, j))
            if d is not None and d.metadata.get("article_id") == section:
                out.append(d)
        return out or [doc]

    def dense_faiss(self, question: str, k: int) -> List[Document]:
        self._refresh()
        if self._faiss is None:
            return []
        return self._faiss.similarity_search(question, k=k)

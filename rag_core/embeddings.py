"""E5 embeddings as a LangChain Embeddings object.

E5 models expect "query: " in front of questions and "passage: " in front of
documents. Wrapping the SentenceTransformer here keeps those prefixes in one place.
"""
from typing import List

from langchain_core.embeddings import Embeddings


class E5Embeddings(Embeddings):
    def __init__(self, model):
        self.model = model  # a sentence_transformers.SentenceTransformer

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode([f"passage: {t}" for t in texts], batch_size=8)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(f"query: {text}").tolist()

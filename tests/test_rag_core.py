import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from langchain_core.documents import Document  # noqa: E402

from rag_core.retrieval import rerank, rrf_fuse  # noqa: E402
from rag_core.text import REFUSAL_MESSAGE, french_tokenize, parse_answer  # noqa: E402


def doc(i):
    return Document(page_content=f"chunk {i}", metadata={"source_document": "a.pdf", "chunk_index": i})


def test_rrf_rewards_documents_found_by_both_retrievers():
    dense = [doc(1), doc(2), doc(3)]
    bm25 = [doc(3), doc(4), doc(1)]
    fused = [d.metadata["chunk_index"] for d in rrf_fuse([dense, bm25])]
    assert fused[:2] == [1, 3]           # in both lists, near the top
    assert set(fused) == {1, 2, 3, 4}    # nothing lost, no duplicates
    assert len(fused) == 4


def test_rerank_keeps_best_top_k():
    class R:
        def predict(self, pairs):
            return [len(d) for _, d in pairs]
    docs = [Document(page_content="a"), Document(page_content="aaa"), Document(page_content="aa")]
    assert [d.page_content for d in rerank(R(), "q", docs, 2)] == ["aaa", "aa"]


def test_french_tokenize_drops_accents_and_stopwords():
    assert french_tokenize("Quelles sont les règles de sécurité ?") == ["regles", "securite"]


def test_parse_answer_full_refusal_partial_answer_and_untagged_refusal():
    assert parse_answer(f"[HORS_CORPUS] {REFUSAL_MESSAGE}") == (REFUSAL_MESSAGE, True)
    answer, refused = parse_answer(f"Selon la FICHE 4, ... [HORS_CORPUS] {REFUSAL_MESSAGE}")
    assert refused is False and "[HORS_CORPUS]" not in answer
    assert parse_answer(REFUSAL_MESSAGE)[1] is True

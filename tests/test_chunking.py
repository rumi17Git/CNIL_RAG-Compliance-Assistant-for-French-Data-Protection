import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion_service"))
import chunking  # noqa: E402

PDF = os.path.join(os.path.dirname(__file__), "..", "pdfs", "cnil_guide_securite_personnelle_2024.pdf")


def load_chunks():
    with open(PDF, "rb") as f:
        return chunking.parse_pdf(f.read(), "cnil_guide_securite_personnelle_2024.pdf")


def test_clean_text_keeps_heading_glued_to_page_number():
    assert chunking.clean_text("5FICHE 1 - PILOTER LA SÉCURITÉ", 5) == "FICHE 1 - PILOTER LA SÉCURITÉ"


def test_clean_text_removes_soft_hyphens_and_control_chars():
    assert chunking.clean_text("sécu\xadrité\x07des", 1) == "sécurité des"


def test_all_25_fiches_are_found():
    fiches = {c["fiche"] for c in load_chunks() if c["fiche"] is not None}
    assert fiches == set(range(1, 26))


def test_chunks_stay_small():
    assert max(len(c["text_content"]) for c in load_chunks()) <= chunking.CHUNK_SIZE + 150


def test_every_chunk_starts_with_its_section_title():
    for c in load_chunks():
        assert c["text_content"].startswith(c["article_id"])


def test_chunking_is_deterministic():
    a = [(c["chunk_index"], c["text_content"]) for c in load_chunks()]
    b = [(c["chunk_index"], c["text_content"]) for c in load_chunks()]
    assert a == b


def test_page_numbers_are_in_range():
    for c in load_chunks():
        assert 1 <= c["page_number"] <= c["page_end"] <= 64

"""
PDF parsing and chunking for the CNIL guide.

Two passes:
  1. Structure: split the document into sections on the guide's own headings
     ("FICHE 4 - AUTHENTIFIER LES UTILISATEURS", ...). Text before the first
     fiche goes into an "AVANT-PROPOS" section.
  2. Size: split each section into chunks of about CHUNK_SIZE characters with a
     recursive splitter, so every chunk fits in the embedding model's
     512-token window.

Each chunk starts with its section title, which helps retrieval and gives a
readable citation (article_id).
"""

import io
import re
from typing import Dict, List

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 900       # characters; about 250-300 tokens of French text
CHUNK_OVERLAP = 120

# "FICHE 4 - TITRE", "FICHE 16- TITRE", "FICHE 3 – TITRE". PyPDF2 often glues the
# heading to the footnotes printed before it, so it is not anchored to a line start;
# the title must be in capitals and runs to the end of the line.
HEADING_RE = re.compile(r"(?m)FICHE[ \t]*(\d{1,2})[ \t]*[-–][ \t]*([A-ZÀ-ÖØ-Þ][^\n]*?)[ \t]*$")
# A short all-caps line right after a heading continues the title
TITLE_CONT_RE = re.compile(r"^[ \t]*([A-ZÀ-ÖØ-Þ0-9’' ,:\-–]{3,60})[ \t]*$")
# Table of contents entries ("FICHE N° 12")
TOC_RE = re.compile(r"FICHE\s+N°\s*\d+")
# Running header of the closing self-assessment pages
ANNEX_RE = re.compile(r"(?m)^[ \t]*FICHES MESURES[ \t]*\n?")
ANNEX_TITLE = "ANNEXE - FICHES MESURES : ÉVALUER LE NIVEAU DE SÉCURITÉ"


def clean_text(text: str, page_number: int = None) -> str:
    """Remove the page number, soft hyphens and control characters; normalise spaces.

    PyPDF2 glues the page number to the first word of the page ("5FICHE 1 - ..."),
    so it is removed using the known page number instead of a pattern that could
    also eat the start of a heading.
    """
    text = text.lstrip()
    if page_number is not None and text.startswith(str(page_number)):
        text = text[len(str(page_number)):]
    text = text.replace("\xad", "")                           # soft hyphens
    text = text.replace("\x07", " ")                          # used as spaces in footnotes
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*\n\s*", "\n\n", text)
    return text.strip()


def _normalise_title(number: str, title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip(" -–")
    return f"FICHE {int(number)} - {title}"


def split_into_sections(pages: List[str]) -> List[Dict]:
    """
    Group cleaned page texts into sections, one per fiche.
    Returns a list of {"title", "fiche", "text", "page_starts"} where
    page_starts is a list of (char_offset_in_section, page_number).
    """
    sections: List[Dict] = []
    current = {"title": "AVANT-PROPOS", "fiche": None, "text": "", "page_starts": []}

    def add(section, page_number, text):
        if not text.strip():
            return
        if section["text"]:
            section["text"] += "\n\n"
        section["page_starts"].append((len(section["text"]), page_number))
        section["text"] += text.strip()

    for page_number, page_text in enumerate(pages, start=1):
        if not page_text or len(TOC_RE.findall(page_text)) >= 5:
            continue  # empty page or table of contents

        if ANNEX_RE.search(page_text):
            page_text = ANNEX_RE.sub("", page_text)
            if current["title"] != ANNEX_TITLE:
                sections.append(current)
                current = {"title": ANNEX_TITLE, "fiche": None, "text": "", "page_starts": []}

        pos = 0
        for match in HEADING_RE.finditer(page_text):
            number = int(match.group(1))
            if match.group(2) != match.group(2).upper() or (current["fiche"] or 0) >= number:
                continue  # not a heading: lower-case title or a reference back to an earlier fiche
            add(current, page_number, page_text[pos:match.start()])
            sections.append(current)

            title = match.group(2)
            rest = page_text[match.end():]
            next_line = rest.lstrip("\n").split("\n", 1)[0]
            cont = TITLE_CONT_RE.match(next_line)
            body_start = match.end()
            if cont and next_line.strip():
                title = f"{title} {cont.group(1)}"
                body_start = match.end() + rest.index(next_line) + len(next_line)

            current = {"title": _normalise_title(match.group(1), title),
                       "fiche": int(match.group(1)), "text": "", "page_starts": []}
            pos = body_start
        add(current, page_number, page_text[pos:])

    sections.append(current)
    return [s for s in sections if s["text"].strip()]


def _page_at(offset: int, page_starts) -> int:
    page = page_starts[0][1]
    for start, number in page_starts:
        if start <= offset:
            page = number
        else:
            break
    return page


def chunk_sections(sections: List[Dict], filename: str,
                   chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks: List[Dict] = []
    for section in sections:
        for doc in splitter.create_documents([section["text"]]):
            start = doc.metadata["start_index"]
            end = start + len(doc.page_content)
            chunks.append({
                "source_document": filename,
                "article_id": section["title"],
                "fiche": section["fiche"],
                "page_number": _page_at(start, section["page_starts"]),
                "page_end": _page_at(max(start, end - 1), section["page_starts"]),
                "chunk_index": len(chunks),
                "text_content": f"{section['title']}\n{doc.page_content.strip()}",
            })
    return chunks


def parse_pdf(file_bytes: bytes, filename: str) -> List[Dict]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [clean_text(page.extract_text() or "", n) for n, page in enumerate(reader.pages, start=1)]
    return chunk_sections(split_into_sections(pages), filename)

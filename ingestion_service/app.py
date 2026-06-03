from fastapi import FastAPI, UploadFile, File
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import os
import io
import re
import uuid
from PyPDF2 import PdfReader

app = FastAPI()

# ---- Configuration ----
qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
collection_name = os.getenv("COLLECTION_NAME", "cnil_docs")
model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
embedder = SentenceTransformer(model_name, device="cpu")
embedding_dim = 768

qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)

# ---- Helper: clean text (remove page headers/footers, noise) ----
def _clean_text(text: str) -> str:
    # Remove leading page numbers (e.g., "61\n" or "55 ")
    text = re.sub(r"^\d{1,3}\n", "", text)
    # Remove patterns like "61FICHES MESURES" or "55FICHE 23"
    text = re.sub(r"\d{1,3}FICHE[S]?\s*(MESURES)?", "", text)
    # Remove soft hyphens (Unicode \xad)
    text = re.sub(r"\xad", "", text)
    # Normalize multiple spaces and line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*\n\s*", "\n\n", text)
    return text.strip()

# ---- Helper: parse PDF into chunks with cleaning ----
def parse_pdf(file_bytes: bytes, filename: str):
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks = []
    article_pattern = r"(Article\s+[\d\-–]+)"
    chunk_idx = 0

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text()
        if not raw_text:
            continue
        # Clean the whole page text first
        text = _clean_text(raw_text)
        if not text:
            continue

        parts = re.split(article_pattern, text)
        if len(parts) == 1:
            chunks.append({
                "source_document": filename,
                "page_number": page_num,
                "article_id": None,
                "chunk_index": chunk_idx,
                "text_content": text
            })
            chunk_idx += 1
        else:
            for i in range(1, len(parts), 2):
                article_title = _clean_text(parts[i])
                article_content = _clean_text(parts[i+1]) if i+1 < len(parts) else ""
                full_chunk = f"{article_title}\n{article_content}".strip()
                if full_chunk:
                    chunks.append({
                        "source_document": filename,
                        "page_number": page_num,
                        "article_id": article_title,
                        "chunk_index": chunk_idx,
                        "text_content": full_chunk
                    })
                    chunk_idx += 1
    return chunks

# ---- Qdrant init ----
@app.on_event("startup")
def init_qdrant():
    try:
        qdrant.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception:
        print(f"Creating collection '{collection_name}' with dimension {embedding_dim}...")
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE)
        )

@app.get("/ping")
def ping():
    return {"message": "Ingestion service is alive"}

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        return {"error": "Only PDF files are accepted"}

    contents = await file.read()
    chunks = parse_pdf(contents, file.filename)

    if not chunks:
        return {"filename": file.filename, "error": "No text extracted"}

    # Prepare texts for embedding
    texts = [chunk["text_content"] for chunk in chunks]
    prefixed_texts = [f"passage: {t}" for t in texts]

    # Encode
    embeddings = embedder.encode(prefixed_texts, batch_size=4, show_progress_bar=True)

    # Upsert to Qdrant
    points = []
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={
                    "source_document": chunk["source_document"],
                    "page_number": chunk["page_number"],
                    "article_id": chunk["article_id"],
                    "chunk_index": chunk["chunk_index"],
                    "text_content": chunk["text_content"]   # already cleaned
                }
            )
        )
    qdrant.upsert(collection_name=collection_name, points=points)

    return {
        "filename": file.filename,
        "chunks_created": len(chunks),
        "status": "embedded and stored in Qdrant"
    }
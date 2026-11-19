import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from chunking import parse_pdf

# ---- Configuration ----
qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
collection_name = os.getenv("COLLECTION_NAME", "cnil_docs")
model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

embedder = SentenceTransformer(model_name, device="cpu")
embedding_dim = embedder.get_sentence_embedding_dimension()
max_tokens = embedder.max_seq_length  # 512 for the E5 models

qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)

# Fixed namespace so the same chunk always gets the same id
POINT_NAMESPACE = uuid.UUID("6f1c2a4e-3b7d-4c9a-8e21-5d0f9b7a1c33")


def point_id(source_document: str, chunk_index: int) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{source_document}:{chunk_index}"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not qdrant.collection_exists(collection_name):
        print(f"Creating collection '{collection_name}' with dimension {embedding_dim}...")
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE),
        )
    else:
        size = qdrant.get_collection(collection_name).config.params.vectors.size
        if size != embedding_dim:
            raise RuntimeError(
                f"Collection '{collection_name}' has vectors of size {size}, but {model_name} "
                f"produces {embedding_dim}. Run 'make clean' or use another COLLECTION_NAME."
            )
        print(f"Collection '{collection_name}' already exists.")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
def ping():
    return {"message": "Ingestion service is alive"}


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    contents = await file.read()
    chunks = parse_pdf(contents, file.filename)
    if not chunks:
        raise HTTPException(status_code=422, detail="No text extracted from the PDF")

    prefixed_texts = [f"passage: {c['text_content']}" for c in chunks]

    # Check that every chunk fits in the model's window (anything longer is cut off)
    token_counts = [len(embedder.tokenizer(t)["input_ids"]) for t in prefixed_texts]
    over_limit = [i for i, n in enumerate(token_counts) if n > max_tokens]

    embeddings = embedder.encode(prefixed_texts, batch_size=8, show_progress_bar=True)

    # Replace this document's old chunks, so re-ingesting never leaves duplicates
    # or leftovers from an older chunking
    qdrant.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="source_document", match=models.MatchValue(value=file.filename))
        ])),
    )
    points = [
        models.PointStruct(
            id=point_id(chunk["source_document"], chunk["chunk_index"]),
            vector=emb.tolist(),
            payload=chunk,
        )
        for chunk, emb in zip(chunks, embeddings)
    ]
    qdrant.upsert(collection_name=collection_name, points=points)

    return {
        "filename": file.filename,
        "chunks_created": len(chunks),
        "sections": len({c["article_id"] for c in chunks}),
        "max_tokens_per_chunk": max(token_counts),
        "chunks_over_token_limit": len(over_limit),
        "status": "embedded and stored in Qdrant",
    }

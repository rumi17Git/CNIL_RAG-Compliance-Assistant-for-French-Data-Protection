from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import requests
from typing import List, Optional
from groq import Groq

app = FastAPI()

# Configuration
qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
collection_name = os.getenv("COLLECTION_NAME", "cnil_docs")
model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")

# Load embedding model (CPU)
embedder = SentenceTransformer(model_name, device="cpu")

# Load cross‑encoder reranker (small, fast, multilingual)
reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', device="cpu")

# Groq client
groq_client = Groq(api_key=groq_api_key)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class Citation(BaseModel):
    text: str
    source_document: str
    page_number: int
    article_id: Optional[str]

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]

@app.get("/ping")
def ping():
    return {"message": "Generation service is alive"}

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    # 1. Embed question with "query:" prefix
    query_vector = embedder.encode(f"query: {req.question}").tolist()

    # 2. Retrieve more candidates than needed (for reranking)
    initial_k = max(req.top_k * 3, 15)  # fetch at least 15
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}/collections/{collection_name}/points/search"
    search_payload = {
        "vector": query_vector,
        "limit": initial_k,
        "with_payload": True
    }
    response = requests.post(qdrant_url, json=search_payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Qdrant search failed")
    search_result = response.json()["result"]

    if not search_result:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    # 3. Rerank using cross‑encoder
    pairs = [[req.question, point["payload"]["text_content"]] for point in search_result]
    scores = reranker.predict(pairs)
    # Combine and sort by score (higher is better)
    scored = sorted(zip(search_result, scores), key=lambda x: x[1], reverse=True)
    final_points = [point for point, _ in scored[:req.top_k]]

    # 4. Build context and citations from reranked results
    contexts = []
    citations = []
    for point in final_points:
        payload = point["payload"]
        chunk_text = payload.get("text_content", "")
        contexts.append(chunk_text)
        citations.append(Citation(
            text=chunk_text,
            source_document=payload.get("source_document", "unknown"),
            page_number=payload.get("page_number", 0),
            article_id=payload.get("article_id")
        ))

    context_str = "\n\n---\n\n".join(contexts)

    # 5. French strict grounding prompt
    prompt = f"""Tu es un expert en conformité CNIL (Commission Nationale de l'Informatique et des Libertés). 
Réponds UNIQUEMENT à partir du contexte ci-dessous. Si la réponse ne s'y trouve pas, dis : 
"Je ne trouve pas cette information dans les documents fournis."

CONTEXTE :
{context_str}

QUESTION : {req.question}

RÉPONSE (en français, avec citations explicites des articles si disponibles) :"""

    # 6. Call Groq (LLaMA 3.1 8B)
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    answer = completion.choices[0].message.content

    return QueryResponse(answer=answer, citations=citations)
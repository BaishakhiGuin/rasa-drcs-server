import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from docx import Document as DocxDocument
from chonkie import SlumberChunker, QdrantHandshake, Model2VecEmbeddings
from chonkie.genie import GroqGenie
from groq import Groq
import tempfile

GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
QDRANT_URL     = "https://f9aedf04-256b-465a-bd31-2849d08c7564.eu-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QA_MODEL       = "qwen/qwen3-32b"
CHUNK_MODEL    = "qwen/qwen3-32b"

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.options("/{rest_of_path:path}")
async def preflight(rest_of_path: str, request: Request):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

# ── Lazy globals (loaded on first request, not at startup) ────
_embedder    = None
_groq_client = None
handshakes   = {}

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = Model2VecEmbeddings("minishlab/potion-base-32M")
    return _embedder

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client

def get_or_create_handshake(collection: str) -> QdrantHandshake:
    if collection not in handshakes:
        handshakes[collection] = QdrantHandshake(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=collection,
            embedding_model=get_embedder(),
        )
    return handshakes[collection]

def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "docx":
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        doc = DocxDocument(tmp_path)
        os.unlink(tmp_path)
        return "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    elif ext in ("txt", "md"):
        return file_bytes.decode("utf-8", errors="replace")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    collection: str = "rag_collection",
    chunk_size: int = 512,
):
    file_bytes = await file.read()
    text = extract_text(file_bytes, file.filename)

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        client.delete_collection(collection)
    except Exception:
        pass

    genie = GroqGenie(model=CHUNK_MODEL, api_key=GROQ_API_KEY)
    chunker = SlumberChunker(genie=genie, chunk_size=chunk_size, verbose=False)
    chunks = chunker(text)

    hs = QdrantHandshake(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection,
        embedding_model=get_embedder(),
    )
    hs.write(chunks)
    handshakes[collection] = hs

    return {
        "status": "ok",
        "collection": collection,
        "chunks": len(chunks),
        "characters": len(text),
        "preview": [
            {"id": i, "text": c.text[:120], "tokens": c.token_count}
            for i, c in enumerate(chunks[:5])
        ],
    }


class AskRequest(BaseModel):
    question: str
    collection: str = "rag_collection"
    top_k: int = 5


@app.post("/ask")
def ask(req: AskRequest):
    hs = get_or_create_handshake(req.collection)
    results = hs.search(query=req.question, limit=req.top_k)

    context_parts = []
    sources = []
    for i, r in enumerate(results):
        text = r.get("text") or r.get("chunk", {}).get("text", "")
        score = r.get("score", 0)
        if text:
            context_parts.append(f"[Chunk {i+1} | score={score:.3f}]\n{text}")
            sources.append({"index": i+1, "score": round(score, 3), "preview": text[:100]})

    if not context_parts:
        return {"answer": "No relevant content found in the document.", "sources": []}

    context = "\n\n---\n\n".join(context_parts)
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I don't have that information in the document."
Be concise and direct.

CONTEXT:
{context}

QUESTION: {req.question}

ANSWER:"""

    response = get_groq_client().chat.completions.create(
        model=QA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": sources,
    }


@app.get("/collections")
def list_collections():
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        cols = client.get_collections()
        return {"collections": [c.name for c in cols.collections]}
    except Exception as e:
        return {"collections": [], "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

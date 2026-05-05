from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import store
from .config import settings
from .ingest import ingest_pdf
from .llm import ollama
from .retrieve import answer

app = FastAPI(title="Private AI Document Assistant", version="0.4.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = None


@app.get("/health")
def health() -> dict:
    try:
        return ollama.health()
    except Exception as e:
        return {"ollama": "unreachable", "error": str(e)}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only .pdf files are supported")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    return ingest_pdf(file.filename, raw)


@app.get("/documents")
def documents() -> dict:
    return {"documents": store.list_sources()}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return answer(req.question, top_k=req.top_k)


@app.delete("/documents/{filename}")
def delete_document(filename: str) -> dict:
    if "/" in filename or "\\" in filename or filename in ("", ".", ".."):
        raise HTTPException(status_code=400, detail="invalid filename")
    store.delete_by_source(filename)
    upload_path = Path(settings.upload_dir) / filename
    upload_path.unlink(missing_ok=True)
    return {"deleted": filename}

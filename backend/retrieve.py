from . import store
from .config import settings
from .llm import ollama


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    k = top_k or settings.top_k
    [embedding] = ollama.embed([question])
    res = store.query(embedding, top_k=k)

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits: list[dict] = []
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append(
            {
                "source": meta.get("source"),
                "page": meta.get("page"),
                "chunk_index": meta.get("chunk_index"),
                "text": doc,
                "distance": dist,
            }
        )
    return hits


SYSTEM_PROMPT = (
    "You are a document assistant. Answer the user's question using the CONTEXT below. "
    "The CONTEXT is your only source of truth — do not use outside knowledge. "
    "If the CONTEXT clearly does not contain enough information, say: "
    "\"I don't know based on the provided documents.\" "
    "Otherwise, answer directly and concisely from the CONTEXT. "
    "When useful, cite sources inline like [source: filename, page: N]."
)


def build_prompt(question: str, hits: list[dict]) -> str:
    if not hits:
        return f"CONTEXT:\n(no documents indexed)\n\nQUESTION: {question}\n\nANSWER:"
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] (source: {h['source']}, page: {h['page']})\n{h['text']}"
        )
    context = "\n\n".join(blocks)
    return f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"


def answer(question: str, top_k: int | None = None) -> dict:
    hits = retrieve(question, top_k=top_k)
    prompt = build_prompt(question, hits)
    text = ollama.generate(prompt=prompt, system=SYSTEM_PROMPT)
    return {"answer": text.strip(), "sources": hits}

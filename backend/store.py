import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings

_client = chromadb.PersistentClient(
    path=settings.chroma_dir,
    settings=ChromaSettings(anonymized_telemetry=False),
)

collection = _client.get_or_create_collection(
    name=settings.collection_name,
    metadata={"hnsw:space": "cosine"},
)


def add_chunks(
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> None:
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def delete_by_source(source: str) -> None:
    collection.delete(where={"source": source})


def list_sources() -> list[dict]:
    res = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in res.get("metadatas") or []:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return [{"source": s, "chunks": c} for s, c in sorted(counts.items())]


def query(embedding: list[float], top_k: int) -> dict:
    return collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

import httpx

from .config import settings


class OllamaClient:
    def __init__(self, host: str = settings.ollama_host):
        self.host = host.rstrip("/")
        self._client = httpx.Client(timeout=120.0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        r = self._client.post(
            f"{self.host}/api/embed",
            json={"model": settings.embed_model, "input": texts},
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload = {
            "model": settings.llm_model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        r = self._client.post(f"{self.host}/api/generate", json=payload)
        r.raise_for_status()
        return r.json()["response"]

    def health(self) -> dict:
        r = self._client.get(f"{self.host}/api/tags")
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return {
            "ollama": "ok",
            "models": models,
            "embed_model_present": any(settings.embed_model in m for m in models),
            "llm_model_present": any(settings.llm_model in m for m in models),
        }


ollama = OllamaClient()

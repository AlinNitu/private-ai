# Private AI Document Assistant (MVP)

Local RAG over your own PDFs. FastAPI + ChromaDB + Ollama, Streamlit UI.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

## One-time setup

```bash
# 1. install Ollama models
ollama pull nomic-embed-text
ollama pull llama3.2:3b

# 2. clone / cd into the project
cd private-ai

# 3. create venv + install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. config
cp .env.example .env
```

## Run (Step 1 — backend smoke test)

In one terminal, make sure Ollama is running:

```bash
ollama serve   # usually already running as a background service
```

In another:

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000
```

Then verify the backend can reach Ollama and that both models are pulled:

```bash
curl http://localhost:8000/health
```

Expected (something like):

```json
{
  "ollama": "ok",
  "models": ["llama3.2:3b", "nomic-embed-text:latest"],
  "embed_model_present": true,
  "llm_model_present": true
}
```

If `embed_model_present` or `llm_model_present` is false, run the `ollama pull` commands above.

## Run the UI (Step 4)

Keep `uvicorn` running in one terminal. In a **second terminal**:

```bash
source .venv/bin/activate
streamlit run frontend/app.py
```

Streamlit opens at http://localhost:8501. From there:

1. Sidebar → **Upload PDF** → pick a file → **Index this PDF**
2. Type a question → **Ask**
3. The answer appears with expandable **Sources** (each shows page number, cosine distance, and the chunk text)

## API endpoints

- `GET  /health` — Ollama reachability + which models are available
- `POST /upload` — multipart form, field name `file`, must be `.pdf`
- `GET  /documents` — list indexed sources and chunk counts
- `POST /ask` — JSON `{ "question": "...", "top_k": 4 }`

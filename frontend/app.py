import httpx
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(page_title="Private AI", layout="wide")
st.title("Private AI Document Assistant")


def fetch_documents() -> list[dict]:
    try:
        r = httpx.get(f"{API}/documents", timeout=30)
        r.raise_for_status()
        return r.json().get("documents", [])
    except Exception as e:
        st.error(f"Cannot reach backend at {API}: {e}")
        return []


def upload_pdf(name: str, data: bytes) -> dict | None:
    files = {"file": (name, data, "application/pdf")}
    try:
        r = httpx.post(f"{API}/upload", files=files, timeout=300)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None


def delete_document(filename: str) -> bool:
    try:
        r = httpx.delete(f"{API}/documents/{filename}", timeout=60)
        r.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False


def ask(question: str, top_k: int) -> dict | None:
    try:
        r = httpx.post(
            f"{API}/ask",
            json={"question": question, "top_k": top_k},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Ask failed: {e}")
        return None


with st.sidebar:
    st.header("Documents")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)
    if uploaded is not None and st.button("Index this PDF", type="primary"):
        with st.spinner(f"Indexing {uploaded.name}…"):
            info = upload_pdf(uploaded.name, uploaded.getvalue())
        if info is not None:
            if info.get("chunks", 0) == 0:
                st.warning(
                    f"No text extracted from {info['filename']} "
                    "(possibly a scanned PDF)."
                )
            else:
                st.success(
                    f"Indexed **{info['filename']}** — "
                    f"{info['pages']} pages, {info['chunks']} chunks."
                )

    st.divider()

    docs = fetch_documents()
    if not docs:
        st.caption("No documents indexed yet.")
    else:
        for d in docs:
            col_info, col_btn = st.columns([4, 1])
            with col_info:
                st.write(f"**{d['source']}** — {d['chunks']} chunks")
            with col_btn:
                if st.button("Delete", key=f"del_{d['source']}"):
                    if delete_document(d["source"]):
                        st.session_state.pop("last_result", None)
                        st.rerun()

st.subheader("Ask a question")
question = st.text_area(
    "Question",
    placeholder="What is this document about?",
    height=100,
    label_visibility="collapsed",
)
top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=4)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Thinking…"):
        result = ask(question.strip(), top_k)
    if result is not None:
        st.session_state["last_result"] = result

result = st.session_state.get("last_result")
if result:
    st.markdown("### Answer")
    st.write(result["answer"])

    sources = result.get("sources", [])
    st.markdown(f"### Sources ({len(sources)})")
    for i, s in enumerate(sources, start=1):
        label = (
            f"{i}. {s['source']} — page {s['page']}  "
            f"(distance: {s['distance']:.3f})"
        )
        with st.expander(label):
            st.text(s["text"])

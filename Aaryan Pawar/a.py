import os
import shutil
import io
import json
import streamlit as st
import pytesseract
import requests
from pdf2image import convert_from_bytes
from PIL import Image

# Optional: try to import sklearn for better similarity search
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

pytesseract.pytesseract.tesseract_cmd = r"F:\\tesseract ocr\\tesseract.EXE"

POPPLER_PATH = r"F:\\poppler-25.07.0\\Library\\bin"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama2"  # change to mistral/qwen/etc. if installed


def _check_dependencies():
    msgs = []
    tesseract_path = shutil.which("tesseract") or pytesseract.pytesseract.tesseract_cmd
    poppler_path = shutil.which("pdftoppm") or os.path.join(POPPLER_PATH, "pdftoppm.EXE")
    if tesseract_path:
        msgs.append(f"Tesseract found at: {tesseract_path}")
    else:
        msgs.append("Tesseract not found in PATH.")
    if poppler_path:
        msgs.append(f"Poppler (pdftoppm) found at: {poppler_path}")
    else:
        msgs.append("Poppler (pdftoppm) not found in PATH.")
    return (bool(tesseract_path), bool(poppler_path), msgs)


def ocr_pdf_to_text(uploaded_file, dpi=300, fmt="jpeg"):
    pdf_bytes = uploaded_file.getvalue()
    images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt=fmt, poppler_path=POPPLER_PATH)
    extracted_text_list = []
    for i, page_image in enumerate(images):
        if page_image.mode != "RGB":
            page_image = page_image.convert("RGB")
        text = pytesseract.image_to_string(page_image)
        extracted_text_list.append({"page_number": i + 1, "text_content": text})
    return extracted_text_list


def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    if not text:
        return []
    chunks, start, text_len = [], 0, len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build_chunks_from_pages(extracted_pages, chunk_size=1000, overlap=200):
    chunks, cid = [], 0
    for p in extracted_pages:
        page_no, page_text = p["page_number"], p["text_content"] or ""
        page_chunks = split_text_into_chunks(page_text, chunk_size=chunk_size, overlap=overlap)
        for ch in page_chunks:
            cid += 1
            chunks.append({"id": cid, "page_number": page_no, "chunk_text": ch.strip()})
    return chunks


def naive_score_query_to_chunks(query, chunks, top_k=3):
    q_terms = [t.lower() for t in query.split() if t.strip()]
    scores = []
    for c in chunks:
        text = c["chunk_text"].lower()
        score = sum(text.count(t) for t in q_terms)
        scores.append(score)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ranked_idx[:top_k] if scores[i] > 0]


def sklearn_retrieve(query, chunks, top_k=3):
    texts = [c["chunk_text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf).flatten()
    ranked_idx = sims.argsort()[::-1]
    top = []
    for i in ranked_idx[:top_k]:
        if sims[i] > 0:
            c = chunks[i].copy()
            c["score"] = float(sims[i])
            top.append(c)
    return top


def ask_ollama(question, context):
    prompt = (
        "You are a helpful assistant. Answer the following question using the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("response") or data.get("message") or str(data)
        else:
            return f"[Ollama error {resp.status_code}] {resp.text}"
    except Exception as e:
        return f"[Ollama request failed] {e}"


# --- Streamlit UI ---
st.set_page_config(page_title="PDF OCR + Chat", layout="wide")
st.title("Public Policy Navigation using AI — OCR + Ollama Chat")

# Dependency checks
t_ok, p_ok, dep_msgs = _check_dependencies()
for m in dep_msgs:
    st.caption(m)
if not t_ok or not p_ok:
    st.warning("Tesseract and/or Poppler may be missing. Install them and ensure their executables are accessible.")

st.markdown("Upload a PDF, extract text, split into chunks, and chat with Ollama about it.")

# Sidebar options
with st.sidebar:
    st.header("Settings")
    dpi = st.number_input("DPI", min_value=100, max_value=600, value=300, step=50)
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=5000, value=1200, step=100)
    overlap = st.number_input("Overlap", min_value=0, max_value=1000, value=200, step=50)
    top_k = st.number_input("Top K chunks", min_value=1, max_value=10, value=3)
    st.caption(f"Similarity engine: {'sklearn TF-IDF' if SKLEARN_AVAILABLE else 'naive word overlap'}")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if "extracted_data" not in st.session_state:
    st.session_state["extracted_data"] = None
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")
    if st.button("Start OCR and Chunking"):
        if not t_ok:
            st.error("Tesseract not found.")
        elif not p_ok:
            st.error("Poppler not found.")
        else:
            with st.spinner("Processing..."):
                try:
                    extracted_data = ocr_pdf_to_text(uploaded_file, dpi=dpi)
                    st.session_state["extracted_data"] = extracted_data
                    chunks = build_chunks_from_pages(extracted_data, chunk_size=chunk_size, overlap=overlap)
                    st.session_state["chunks"] = chunks
                    st.success(f"Extracted {len(extracted_data)} pages, {len(chunks)} chunks.")
                except Exception as e:
                    st.error(f"OCR error: {e}")

if st.session_state.get("extracted_data"):
    extracted_data = st.session_state["extracted_data"]
    st.subheader("Preview")
    preview_pages = {p["page_number"]: p["text_content"][:2000] for p in extracted_data[:3]}
    st.json({"preview": preview_pages, "total_pages": len(extracted_data)})

    if st.session_state.get("chunks") and st.checkbox("Show chunks"):
        chunks = st.session_state["chunks"]
        for c in chunks[:5]:
            st.markdown(f"**Chunk {c['id']} (page {c['page_number']})**")
            st.write(c['chunk_text'][:400])

st.markdown("---")
st.subheader("Chat with Ollama")
user_query = st.text_input("Ask a question about the document")

if st.button("Submit") and user_query.strip():
    if not st.session_state.get("chunks"):
        st.error("No chunks available. Run OCR first.")
    else:
        chunks = st.session_state["chunks"]
        if SKLEARN_AVAILABLE:
            top_chunks = sklearn_retrieve(user_query, chunks, top_k=top_k)
        else:
            top_chunks = naive_score_query_to_chunks(user_query, chunks, top_k=top_k)
        context = "\n---\n".join([f"(Page {c['page_number']}) {c['chunk_text']}" for c in top_chunks])
        answer = ask_ollama(user_query, context)

        st.session_state["chat_history"].append({"role": "user", "text": user_query})
        st.session_state["chat_history"].append({"role": "assistant", "text": answer})

st.write("### Conversation")
for msg in st.session_state["chat_history"][-20:]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['text']}")
    else:
        st.markdown(f"**Assistant:** {msg['text']}")

st.caption("Uses Ollama locally. Ensure you have Ollama installed and running a model (e.g., `ollama run llama2`).")
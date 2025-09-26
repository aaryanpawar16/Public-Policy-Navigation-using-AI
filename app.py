# app.py
"""
Full Streamlit File Uploader app (Milestone 3):
- Original styling (cards, hover animations)
- CSV / Excel / TXT / Image / PDF previews
- PDF OCR fallback using pdfplumber + pytesseract (if available)
- Chunking (configurable size & overlap)
- Download chunked text as .txt or JSON, download OCR JSON
- Retrieval: automatic top-N chunk selection for relevance (simple token overlap)
- Chat interface that sends context + user question to local Ollama via HTTP
- Hardcoded model: llama3.1:latest (change HARDCODED_OLLAMA_MODEL if needed)
- Local smarter extractive summarizer fallback when Ollama is unavailable
"""

import io
import time
import json
import math
import re
import string
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import pandas as pd
import streamlit as st
from PIL import Image
import requests

# ---------- Optional OCR / PDF libraries ----------
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False

try:
    import pytesseract
    from pytesseract import image_to_string
    from pdf2image import convert_from_bytes
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

# ---------- Configuration (hardcoded model) ----------
OLLAMA_BASE_URL = "http://localhost:11434"
HARDCODED_OLLAMA_MODEL = "llama3.1:latest"  # change this to exact installed model if different

# ---------- Page config ----------
st.set_page_config(page_title="📂 File Upload App (Hardcoded Ollama)", page_icon="📂", layout="wide")

# ---------- CSS Styling (original) ----------
st.markdown("""
<style>
:root { --card-bg: #0e1117; }
.block-container { padding-top: 2rem; }
hr { border: none; height: 1px; background: #333; }
h2, h3 { letter-spacing: 0.2px; }

.card {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 2px 16px rgba(0,0,0,.25);
    border: 4px solid #2a2f3a;
}

[data-testid="stFileUploader"] section {
    border: 4px solid #3b3f4a;
    border-radius: 12px;
    padding: 6px;
    background: #0e1117;
    transition: all .25s ease;
}
[data-testid="stFileUploader"] section:hover {
    border: 4px solid transparent;
    background:
      linear-gradient(#0e1117, #0e1117) padding-box,
      linear-gradient(30deg, red, orange, yellow, green, blue, indigo, violet) border-box;
    background-size: 300% 300%;
    animation: rainbowShift 3s linear infinite;
}
@keyframes rainbowShift { 0%{background-position:0% 50%} 100%{background-position:100% 50%} }

.stButton>button {
    border: 4px solid #3b3f4a !important;
    border-radius: 12px !important;
    transition: all .25s ease;
}
.stButton>button:hover {
    border: 4px solid transparent !important;
    background:
      linear-gradient(#262730, #262730) padding-box,
      linear-gradient(30deg, red, orange, yellow, green, blue, indigo, violet) border-box !important;
    background-size: 300% 300% !important;
    animation: rainbowShift 3s linear infinite !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.title("📂 File Upload App")
st.write("Upload CSV / Excel / PDF / Image / TXT, preview, chunk, download, and ask questions (Ollama).")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📂 File Uploader")
    st.caption("Milestone — by Isha Garg")
    st.markdown("---")
    st.subheader("Navigation")
    view = st.radio(
        "Choose a view",
        ["Upload & Preview", "Filter & Search (CSV/XLSX)", "About"],
        index=0,
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.subheader("Chunking & Chat config")
    chunk_size = st.number_input("Chunk Size (characters)", 500, 5000, 1000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap (characters)", 0, 1000, 100, step=50)
    max_chunks_for_context = st.slider("Top-N chunks to include (retrieval)", 1, 20, 5)
    st.markdown("---")
    st.subheader("Downloads")
    download_format = st.radio("Download processed text as", ["Chunks (txt)", "JSON (chunks)"], index=1)
    st.markdown("---")
    st.subheader("Ollama Model (HARDCODED)")
    st.markdown(f"- Using hardcoded model: **{HARDCODED_OLLAMA_MODEL}**")
    st.markdown(f"- Ollama URL: **{OLLAMA_BASE_URL}**")
    st.markdown("---")
    st.subheader("Tips")
    st.markdown(
        "- Drag & drop a file\n"
        "- CSV/XLSX → filter, search, download\n"
        "- PDF → pdfplumber primary extraction; pytesseract OCR fallback\n"
        "- Images → preview + metadata\n"
        "- TXT → preview and chunking\n"
        "- Chat: automatic retrieval of top-N chunks will be used as context"
    )
    st.markdown("---")
    st.caption(f"🕒 {datetime.now().strftime('%d %b %Y, %I:%M %p')}")

# ---------- Helper functions ----------
def file_stats(file) -> dict:
    try:
        pos = file.tell()
    except Exception:
        pos = None
    if hasattr(file, "getbuffer"):
        size_kb = round(len(file.getbuffer()) / 1024, 2)
    else:
        data = file.read()
        size_kb = round(len(data) / 1024, 2)
        if pos is not None:
            try:
                file.seek(pos)
            except Exception:
                pass
    return {"name": getattr(file, "name", "uploaded_file"), "type": getattr(file, "type", ""), "size_kb": size_kb}

def simple_text_summary(text: str, max_chars: int = 400) -> str:
    if text is None:
        return ""
    text = " ".join(str(text).split())
    return (text[:max_chars] + "…") if len(text) > max_chars else text

def df_quick_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    return df

def df_info_cards(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='card'><h4>🧾 Rows</h4><h2>{len(df):,}</h2></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><h4>🧭 Columns</h4><h2>{df.shape[1]}</h2></div>", unsafe_allow_html=True)
    with c3:
        num_cols = df.select_dtypes(include="number").shape[1]
        st.markdown(f"<div class='card'><h4>🔢 Numeric cols</h4><h2>{num_cols}</h2></div>", unsafe_allow_html=True)
    with c4:
        obj_cols = df.select_dtypes(include="object").shape[1]
        st.markdown(f"<div class='card'><h4>🔤 Text cols</h4><h2>{obj_cols}</h2></div>", unsafe_allow_html=True)

def chunk_text_chars(text: str, size: int = 1000, overlap: int = 100) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        chunks.append(text[start:end])
        if end == n:
            break
        start += max(1, size - overlap)
    return chunks

def normalize_text_for_retrieval(s: str) -> List[str]:
    tokens = re.findall(r"\w+", s.lower())
    tokens = [t for t in tokens if len(t) > 1]
    return tokens

def score_chunk_by_query(chunk: str, query: str) -> float:
    q_tokens = normalize_text_for_retrieval(query)
    if not q_tokens:
        return 0.0
    c_tokens = normalize_text_for_retrieval(chunk)
    if not c_tokens:
        return 0.0
    from collections import Counter
    qc = Counter(q_tokens)
    cc = Counter(c_tokens)
    score = 0.0
    for term, freq in qc.items():
        score += freq * cc.get(term, 0)
    denom = math.sqrt(len(c_tokens)) if len(c_tokens) > 0 else 1.0
    return score / denom

# ---------- Summarization helpers (smarter local fallback) ----------
# We implement an extractive summarizer:
#  - Sentence tokenization
#  - Term frequency across the document
#  - Sentence scoring = sum(token tf) * position_weight
#  - Optional query boost: sentences containing query tokens are amplified
#  - Keep original order when returning selected sentences

# Minimal stopword list (kept small and local to avoid large dependencies)
_STOPWORDS = {
    "the", "and", "is", "in", "it", "of", "to", "a", "an", "that", "this", "for", "on", "with",
    "as", "are", "was", "were", "by", "be", "or", "from", "at", "which", "has", "have", "not",
    "but", "they", "their", "we", "you", "I", "he", "she", "its", "if", "will", "can", "would"
}

_sentence_splitter_re = re.compile(r'(?<=[.!?])\s+')

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences in a robust but simple way.
    """
    if not text:
        return []
    # Normalize newlines then split
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple newlines
    text = re.sub(r"\n{2,}", "\n\n", text)
    # Split on sentence endings
    parts = _sentence_splitter_re.split(text.strip())
    # Further split long fragments with newlines into logical sentences
    sentences = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # break if extremely long (use punctuation heuristics)
        if len(p) > 800:
            # split on newline or semicolon
            sub = re.split(r'[\n;]+', p)
            for s in sub:
                s = s.strip()
                if s:
                    sentences.append(s)
        else:
            sentences.append(p)
    return sentences

def tokenize_sentence(s: str) -> List[str]:
    s = s.lower()
    # remove punctuation, keep alphanumerics and apostrophes
    s = s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    tokens = [t for t in re.findall(r'\w+', s) if len(t) > 1 and t not in _STOPWORDS]
    return tokens

def build_term_frequencies(sentences: List[str]) -> Dict[str, int]:
    tf = {}
    for sent in sentences:
        for t in tokenize_sentence(sent):
            tf[t] = tf.get(t, 0) + 1
    return tf

def score_sentences(sentences: List[str], query: Optional[str] = None) -> List[float]:
    """
    Score each sentence using term frequency and positional bias.
    If a query is provided, boost sentences containing query tokens.
    """
    tf = build_term_frequencies(sentences)
    total_tokens = sum(tf.values()) or 1
    # normalize tf to avoid extremely large magnitudes
    tf_norm = {k: v / total_tokens for k, v in tf.items()}

    q_tokens = normalize_text_for_retrieval(query) if query else []
    q_set = set(q_tokens)

    scores = []
    n = len(sentences)
    for i, sent in enumerate(sentences):
        tokens = tokenize_sentence(sent)
        # base sentence score: sum normalized tf for tokens (importance in doc)
        base_score = sum(tf_norm.get(t, 0.0) for t in tokens)
        # position weight: earlier sentences slightly more important (but subtle)
        pos_weight = 1.0 + (1.0 - (i / max(1, n - 1))) * 0.15  # between 1.0 and 1.15
        score = base_score * pos_weight
        # query boost
        if q_set:
            match_count = sum(1 for t in tokens if t in q_set)
            if match_count > 0:
                # boost factor depends on how many query tokens sentence contains
                score *= (1.0 + 0.5 * match_count)
        # length penalty for extremely short sentences
        if len(tokens) < 3:
            score *= 0.7
        scores.append(score)
    return scores

def extractive_summary(text: str, max_sentences: int = 6, query: Optional[str] = None) -> str:
    """
    Produce an extractive summary of `text` as up to `max_sentences` sentences.
    If `query` is present, produce a query-focused summary.
    """
    if not text or not text.strip():
        return "📄 No readable text available to summarize."

    # Prepare candidate sentences
    sentences = split_into_sentences(text)
    if not sentences:
        # fallback: return first N characters
        return simple_text_summary(text, max_chars=600)

    # If text is small, return short trimmed version
    joined_len = sum(len(s) for s in sentences)
    if joined_len <= 800:
        # return cleaned short version (trim to reasonable length)
        return " ".join(sentences[:max_sentences])

    # Score sentences
    scores = score_sentences(sentences, query=query)
    # Pair (index, score) and sort by score desc
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)

    # Select top sentences indices
    chosen_idx = [i for i, _ in indexed[:max_sentences]]
    chosen_idx = sorted(set(chosen_idx))
    # Build summary in original order
    summary_sentences = [sentences[i].strip() for i in chosen_idx]
    # Join with spaces and limit char length to ~1000 to be safe
    summary = " ".join(summary_sentences)
    if len(summary) > 1200:
        # Trim preserving sentence boundaries
        out = []
        l = 0
        for s in summary_sentences:
            if l + len(s) > 1200:
                break
            out.append(s)
            l += len(s) + 1
        summary = " ".join(out)
    # Prefix for clarity
    if query:
        prefix = "📄 **Query-focused Local Summary:**\n\n"
    else:
        prefix = "📄 **Brief Local Summary:**\n\n"
    return prefix + summary

def local_fallback_summary(pdf_text: str = "", df: Optional[pd.DataFrame] = None, img: Optional[Image.Image] = None, filename: str = "", query: Optional[str] = None) -> str:
    """
    Generate a local summary if Ollama is unavailable.
    Uses extractive_summary for text; for tables/images returns metadata.
    """
    if pdf_text:
        # If there is a query, create a query-focused summary
        max_sentences = 6 if query else 8
        return extractive_summary(pdf_text, max_sentences=max_sentences, query=query)
    elif df is not None:
        cols_preview = ", ".join(list(df.columns[:8]))
        more = "..." if df.shape[1] > 8 else ""
        return (f"📊 The uploaded file '{filename}' has {len(df):,} rows and {df.shape[1]} columns. "
                f"Columns: {cols_preview}{more}.")
    elif img is not None:
        return f"🖼️ The uploaded image '{filename}' has size {img.size[0]}x{img.size[1]} pixels and mode {img.mode}."
    else:
        return "🤔 I couldn’t extract any textual content to summarize."

# ---------- Ollama HTTP helpers ----------
def ollama_query_http(base_url: str, model: str, prompt: str, timeout: int = 60) -> Tuple[bool, str]:
    headers = {"Content-Type": "application/json"}
    # Try /api/chat first
    chat_payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    try:
        resp = requests.post(base_url.rstrip("/") + "/api/chat", json=chat_payload, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            try:
                data = resp.json()
                if isinstance(data, dict):
                    if "response" in data and isinstance(data["response"], str):
                        return True, data["response"]
                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        if isinstance(choice, dict):
                            msg = choice.get("message", {}).get("content")
                            if msg:
                                return True, msg
                        return True, json.dumps(choice)
                    return True, json.dumps(data)
                return True, str(data)
            except Exception as e:
                return False, f"Could not decode /api/chat response: {e}"
        else:
            try:
                body = resp.json()
                return False, f"/api/chat returned {resp.status_code}: {json.dumps(body)}"
            except Exception:
                return False, f"/api/chat returned {resp.status_code}: {resp.text}"
    except Exception:
        gen_payload = {"model": model, "prompt": prompt}
        try:
            resp2 = requests.post(base_url.rstrip("/") + "/api/generate", json=gen_payload, headers=headers, timeout=timeout)
            if resp2.status_code == 200:
                try:
                    j = resp2.json()
                    if isinstance(j, dict) and "response" in j:
                        return True, j["response"]
                    return True, json.dumps(j)
                except Exception as e:
                    return False, f"Could not decode /api/generate response: {e}"
            else:
                try:
                    body = resp2.json()
                    return False, f"/api/generate returned {resp2.status_code}: {json.dumps(body)}"
                except Exception:
                    return False, f"/api/generate returned {resp2.status_code}: {resp2.text}"
        except Exception as e2:
            return False, f"Could not reach Ollama API at {base_url}: {e2}"

def query_ollama_safe(base_url: str, model: str, prompt: str) -> str:
    ok, out = ollama_query_http(base_url, model, prompt)
    if ok:
        return out
    else:
        return (f"⚠️ Ollama request failed:\n\n{out}\n\n"
                f"Suggested fixes:\n"
                f"- Ensure Ollama is running and accessible at {base_url}\n"
                f"- Ensure the model name matches exactly one installed model (use `ollama list`)\n")

# ---------- File upload ----------
uploaded = st.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "txt", "png", "jpg", "jpeg", "pdf"],
    help="CSV/XLSX, TXT, PNG/JPG, or PDF",
)

pdf_text = ""
chunks: List[str] = []
img: Optional[Image.Image] = None
df: Optional[pd.DataFrame] = None
stats = None

if uploaded:
    stats = file_stats(uploaded)
    st.success("File uploaded!")
    cA, cB, cC = st.columns(3)
    cA.metric("📄 Filename", stats["name"])
    cB.metric("📦 Type", stats["type"] or uploaded.name.split('.')[-1])
    cC.metric("💾 Size (KB)", stats["size_kb"])

# ---------- About ----------
if view == "About":
    st.subheader("About this App")
    st.write("""
    **Smart File Uploader (Milestone 3)** features:
    - Clean UI with animated hover effects
    - File-type aware previews & quick stats
    - Filter & search for CSV/XLSX
    - PDF preview and OCR fallback (pdfplumber + pytesseract)
    - Image and text previews
    - Download extracted text or chunked JSON
    - Chunking + automatic retrieval for Ollama queries
    - Local smarter summarization fallback when Ollama is unavailable
    """)
    st.info("Use the sidebar to switch views.")
    if not uploaded:
        st.stop()

# ---------- Upload & Preview ----------
if view == "Upload & Preview":
    if not uploaded:
        st.info("Upload a file to begin.")
        st.stop()

    uploaded_type = getattr(uploaded, "type", "") or ""
    uploaded_name = uploaded.name.lower()
    pdf_text = ""
    img = None
    df = None
    chunks = []

    with st.spinner("Preparing preview..."):
        time.sleep(0.2)

        # CSV
        if "csv" in uploaded_type or uploaded_name.endswith(".csv"):
            try:
                uploaded.seek(0)
            except Exception:
                pass
            df = pd.read_csv(uploaded)
            df = df_quick_clean(df)
            st.subheader("🔎 CSV Preview")
            st.dataframe(df.head(100), use_container_width=True)
            df_info_cards(df)
            pdf_text = df.to_csv(index=False)

        # Excel
        elif "excel" in uploaded_type or uploaded_name.endswith((".xlsx", ".xls")):
            try:
                uploaded.seek(0)
            except Exception:
                pass
            df = pd.read_excel(uploaded)
            df = df_quick_clean(df)
            st.subheader("🔎 Excel Preview")
            st.dataframe(df.head(100), use_container_width=True)
            df_info_cards(df)
            pdf_text = df.to_csv(index=False)

        # PDF
        elif "pdf" in uploaded_type or uploaded_name.endswith(".pdf"):
            st.subheader("📖 PDF Preview")
            try:
                uploaded.seek(0)
            except Exception:
                pass

            if not HAS_PDFPLUMBER and not HAS_TESSERACT:
                st.warning("Install `pdfplumber` or `pytesseract` + `pdf2image` to enable PDF text preview: pip install pdfplumber pytesseract pdf2image")
            else:
                try:
                    if HAS_PDFPLUMBER:
                        uploaded.seek(0)
                        with pdfplumber.open(uploaded) as pdf:
                            for i, page in enumerate(pdf.pages):
                                try:
                                    text = page.extract_text()
                                except Exception:
                                    text = None
                                if (not text or text.strip() == "") and HAS_TESSERACT:
                                    try:
                                        pil_img = page.to_image(resolution=300).original
                                        page_text = image_to_string(pil_img)
                                    except Exception:
                                        try:
                                            uploaded.seek(0)
                                            images = convert_from_bytes(uploaded.read())
                                            page_text = image_to_string(images[i])
                                        except Exception:
                                            page_text = ""
                                    text = page_text
                                pdf_text += (text or "") + "\n"
                    else:
                        # No pdfplumber, try pdf2image + pytesseract
                        uploaded.seek(0)
                        raw = uploaded.read()
                        images = convert_from_bytes(raw)
                        for img_p in images:
                            pdf_text += image_to_string(img_p) + "\n"

                    if pdf_text.strip():
                        st.text_area("📜 Extracted Text (OCR + PDF)", pdf_text, height=400)
                        pdf_json = {"filename": uploaded.name, "text": pdf_text}
                        json_bytes = json.dumps(pdf_json, indent=2, ensure_ascii=False).encode("utf-8")
                        st.download_button(
                            "💾 Download Extracted Text (JSON)",
                            data=json_bytes,
                            file_name=f"{uploaded.name}_text.json",
                            mime="application/json"
                        )
                    else:
                        st.info("No extractable text found in the PDF.")
                except Exception as e:
                    st.error(f"Could not process PDF: {e}")

        # Image
        elif "image" in uploaded_type or uploaded_name.endswith((".png", ".jpg", ".jpeg")):
            try:
                uploaded.seek(0)
            except Exception:
                pass
            st.subheader("🖼️ Image Preview")
            try:
                img = Image.open(uploaded)
                st.image(img, use_column_width=True)
                w, h = img.size
                c1, c2, c3 = st.columns(3)
                c1.metric("Width", w)
                c2.metric("Height", h)
                c3.metric("Mode", str(img.mode))
                pdf_text = f"Image: {uploaded.name} | size: {w}x{h} | mode: {img.mode}"
            except Exception as e:
                st.error(f"Could not open image: {e}")

        # Text
        elif "text" in uploaded_type or uploaded_name.endswith(".txt"):
            try:
                uploaded.seek(0)
            except Exception:
                pass
            st.subheader("📄 Text Preview")
            try:
                text = uploaded.read().decode("utf-8", errors="ignore")
            except Exception:
                uploaded.seek(0)
                raw = uploaded.getvalue()
                text = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
            pdf_text = text
            st.code(simple_text_summary(text, max_chars=2000), language="markdown")

        else:
            st.warning("This file type is not supported for preview yet.")

    # ---------- Chunking ----------
    if pdf_text:
        chunks = chunk_text_chars(pdf_text, size=chunk_size, overlap=chunk_overlap)
        st.markdown(f"**Processed into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap}).**")
    elif df is not None:
        df_text_preview = df.to_csv(index=False)
        chunks = chunk_text_chars(df_text_preview, size=chunk_size, overlap=chunk_overlap)
        st.markdown(f"**CSV/Excel text representation chunked into {len(chunks)} chunks.**")

    # ---------- Downloads ----------
    if chunks:
        if download_format == "Chunks (txt)":
            chunked_output = "\n\n---\n\n".join(chunks)
            st.download_button(
                "💾 Download Chunked Text (.txt)",
                data=chunked_output,
                file_name=f"{uploaded.name}_chunks.txt",
                mime="text/plain"
            )
        else:
            meta = {"filename": uploaded.name, "size_kb": stats["size_kb"] if stats else None, "chunks": len(chunks)}
            json_out = {"meta": meta, "chunks": chunks}
            json_bytes = json.dumps(json_out, indent=2, ensure_ascii=False).encode("utf-8")
            st.download_button(
                "💾 Download Chunks JSON",
                data=json_bytes,
                file_name=f"{uploaded.name}_chunks.json",
                mime="application/json"
            )

    if pdf_text and (uploaded_name.endswith(".pdf") or uploaded_name.endswith(".txt")):
        ocr_json = {"filename": uploaded.name, "extracted_text": pdf_text}
        ocr_bytes = json.dumps(ocr_json, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button("💾 Download Extracted Text (JSON)", data=ocr_bytes, file_name=f"{uploaded.name}_ocr.json", mime="application/json")

    # ---------- Chat (automatic retrieval, direct answer) ----------
    st.markdown("---")
    st.subheader("💬 Ask a Question about the File (automatic retrieval)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)

    query = st.chat_input("Ask something...")
    if query:
        st.chat_message("user").write(query)
        st.session_state.chat_history.append(("user", query))

        if not pdf_text and df is None and not img:
            response = "🤔 No textual content extracted from the uploaded file to query."
            st.chat_message("assistant").write(response)
            st.session_state.chat_history.append(("assistant", response))
        else:
            # Retrieval: score chunks and select top-N automatically
            if chunks:
                scored = []
                for idx, ch in enumerate(chunks):
                    s = score_chunk_by_query(ch, query)
                    scored.append((idx, s))
                scored.sort(key=lambda x: x[1], reverse=True)
                if scored and scored[0][1] == 0:
                    top_indices = list(range(min(max_chunks_for_context, len(chunks))))
                else:
                    top_indices = [idx for idx, sc in scored[:max_chunks_for_context]]
                top_indices = sorted(set(top_indices))
                chosen_chunks = [chunks[i] for i in top_indices]
                context_text = "\n\n---\n\n".join(chosen_chunks)
            else:
                # if no chunks, pass entire pdf_text up to a limit
                context_text = pdf_text[: max(1, chunk_size * max(1, max_chunks_for_context))]

            # Build prompt including context + question
            full_prompt = (
                "You are a helpful assistant. Use the CONTEXT below to answer the user's question. "
                "If the answer is not contained in the context, say you don't know instead of inventing facts.\n\n"
                f"CONTEXT:\n{context_text}\n\nQUESTION: {query}\n\n"
                "Answer concisely and, if possible, reference which part of the context supports your answer."
            )

            # Query Ollama with hardcoded model
            answer = query_ollama_safe(OLLAMA_BASE_URL, HARDCODED_OLLAMA_MODEL, full_prompt)

            # If Ollama failed, use smarter local summary focused on query
            if answer.startswith("⚠️ Ollama request failed"):
                # query-aware summary fallback
                fallback = local_fallback_summary(pdf_text if pdf_text else None, df, img, uploaded.name, query=query)
                # If we have context chunks, also show which chunks were used
                if chunks:
                    # Provide short explanation of retrieval
                    top_preview = "\n\n---\n\n".join([c[:500] + ("…" if len(c) > 500 else "") for c in chosen_chunks[:3]])
                    answer = (f"{fallback}\n\n"
                              f"---\n"
                              f"⚠️ Ollama unreachable — showing local query-focused summary and top retrieved chunks (truncated):\n\n"
                              f"{top_preview}")
                else:
                    answer = (f"{fallback}\n\n"
                              "⚠️ Ollama unreachable — showing local query-focused summary.")
            # Display and save
            st.chat_message("assistant").write(answer)
            st.session_state.chat_history.append(("assistant", answer))

# ---------- Filter & Search (CSV/XLSX) ----------
if view == "Filter & Search (CSV/XLSX)":
    if not uploaded:
        st.info("Upload a CSV or Excel file to use filter & search.")
        st.stop()

    uploaded_name = uploaded.name.lower()
    if not (uploaded_name.endswith(".csv") or uploaded_name.endswith((".xlsx", ".xls"))):
        st.warning("Filter & Search only works for CSV/XLSX files.")
        st.stop()

    try:
        uploaded.seek(0)
    except Exception:
        pass
    try:
        if uploaded_name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read file as DataFrame: {e}")
        st.stop()

    df = df_quick_clean(df)

    st.subheader("🔎 Filter & Search CSV/XLSX")
    st.write("Use the filters below to refine your data:")

    filter_cols = st.multiselect("Select columns to filter", df.columns.tolist())
    filtered_df = df.copy()

    for col in filter_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val, max_val = float(df[col].min()), float(df[col].max())
            r = st.slider(f"Filter `{col}` range", min_val, max_val, (min_val, max_val))
            filtered_df = filtered_df[(filtered_df[col] >= r[0]) & (filtered_df[col] <= r[1])]
        else:
            vals = df[col].dropna().unique().tolist()
            selected_vals = st.multiselect(f"Filter `{col}` values", vals)
            if selected_vals:
                filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

    search_term = st.text_input("Search in text columns (case-insensitive)")
    if search_term:
        mask = pd.Series(False, index=filtered_df.index)
        for col in filtered_df.select_dtypes(include="object").columns:
            mask = mask | filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)
        filtered_df = filtered_df[mask]

    st.markdown("### 📄 Filtered Data Preview")
    st.dataframe(filtered_df.head(100), use_container_width=True)
    st.markdown(f"**Total rows after filtering:** {len(filtered_df)}")

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Filtered CSV", data=csv_bytes, file_name="filtered_data.csv", mime="text/csv")

# ---------- End of app ----------

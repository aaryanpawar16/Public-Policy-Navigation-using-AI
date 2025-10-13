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
import streamlit as st
import time
import io
import json
import math
import re
import string
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import pandas as pd
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

# ---------- Configuration ----------
OLLAMA_BASE_URL="http://127.0.0.1:11434"
HARDCODED_OLLAMA_MODEL = "llama3.1:latest"  # change to installed model

# ---------- Page config ----------
st.set_page_config(page_title="📂 File Upload App (Hardcoded Ollama)", page_icon="📂", layout="wide")

# ---------- CSS Styling ----------
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

# ---------- Summarization helpers ----------
_STOPWORDS = {
    "the", "and", "is", "in", "it", "of", "to", "a", "an", "that", "this", "for", "on", "with",
    "as", "are", "was", "were", "by", "be", "or", "from", "at", "which", "has", "have", "not",
    "but", "they", "their", "we", "you", "I", "he", "she", "its", "if", "will", "can", "would"
}

_sentence_splitter_re = re.compile(r'(?<=[.!?])\s+')

def split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    parts = _sentence_splitter_re.split(text.strip())
    sentences = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > 800:
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
    tf = build_term_frequencies(sentences)
    total_tokens = sum(tf.values()) or 1
    tf_norm = {k: v / total_tokens for k, v in tf.items()}
    q_tokens = normalize_text_for_retrieval(query) if query else []
    q_set = set(q_tokens)
    scores = []
    n = len(sentences)
    for i, sent in enumerate(sentences):
        tokens = tokenize_sentence(sent)
        base_score = sum(tf_norm.get(t, 0.0) for t in tokens)
        pos_weight = 1.0 + (1.0 - (i / max(1, n - 1))) * 0.15
        score = base_score * pos_weight
        if q_set:
            match_count = sum(1 for t in tokens if t in q_set)
            if match_count > 0:
                score *= (1.0 + 0.5 * match_count)
        if len(tokens) < 3:
            score *= 0.7
        scores.append(score)
    return scores

def extractive_summary(text: str, max_sentences: int = 6, query: Optional[str] = None) -> str:
    if not text or not text.strip():
        return "📄 No readable text available to summarize."
    sentences = split_into_sentences(text)
    if not sentences:
        return simple_text_summary(text, max_chars=600)
    joined_len = sum(len(s) for s in sentences)
    if joined_len <= 800:
        return " ".join(sentences[:max_sentences])
    scores = score_sentences(sentences, query=query)
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    chosen_idx = [i for i, _ in indexed[:max_sentences]]
    chosen_idx = sorted(set(chosen_idx))
    summary_sentences = [sentences[i].strip() for i in chosen_idx]
    summary = " ".join(summary_sentences)
    if len(summary) > 1200:
        out = []
        l = 0
        for s in summary_sentences:
            if l + len(s) > 1200:
                break
            out.append(s)
            l += len(s) + 1
        summary = " ".join(out)
    if query:
        prefix = "📄 **Query-focused Local Summary:**\n\n"
    else:
        prefix = "📄 **Brief Local Summary:**\n\n"
    return prefix + summary

def local_fallback_summary(pdf_text: str = "", df: Optional[pd.DataFrame] = None, img: Optional[Image.Image] = None, filename: str = "", query: Optional[str] = None) -> str:
    if pdf_text:
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
    chat_payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    try:
        resp = requests.post(base_url.rstrip("/") + "/api/chat", json=chat_payload, headers=headers, timeout=120)

        if resp.status_code == 200:
            data = resp.json()
            if "completion" in data:
                return True, data["completion"]
            return True, str(data)
        return False, f"Ollama returned {resp.status_code}: {resp.text[:250]}"
    except Exception as e:
        return False, str(e)

# ---------- Upload & Preview ----------
if view == "Upload & Preview":
    uploaded_file = st.file_uploader("📂 Upload file", type=["csv","xlsx","xls","pdf","txt","png","jpg","jpeg"], accept_multiple_files=False)
    if uploaded_file:
        stats = file_stats(uploaded_file)
        st.markdown(f"**File:** `{stats['name']}` — `{stats['size_kb']} KB` — `{stats['type']}`")
        filename_lower = stats['name'].lower()
        text_content = None
        df = None
        img = None
        # ---------- CSV/XLSX ----------
        if filename_lower.endswith((".csv",".xlsx",".xls")):
            try:
                if filename_lower.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                df = df_quick_clean(df)
                df_info_cards(df)
                st.dataframe(df, height=280)
            except Exception as e:
                st.error(f"Failed to read spreadsheet: {e}")
        # ---------- PDF ----------
        elif filename_lower.endswith(".pdf"):
            pdf_text = ""
            try:
                if HAS_PDFPLUMBER:
                    uploaded_file.seek(0)
                    with pdfplumber.open(uploaded_file) as pdf:
                        pdf_text = "\n".join([p.extract_text() or "" for p in pdf.pages])
                if not pdf_text and HAS_TESSERACT:
                    uploaded_file.seek(0)
                    images = convert_from_bytes(uploaded_file.read())
                    ocr_texts = [image_to_string(im) for im in images]
                    pdf_text = "\n".join(ocr_texts)
                text_content = pdf_text.strip()
                st.text_area("📄 Extracted PDF text preview:", text_content[:3000], height=300)
            except Exception as e:
                st.error(f"PDF extraction failed: {e}")
        # ---------- TXT ----------
        elif filename_lower.endswith(".txt"):
            uploaded_file.seek(0)
            text_content = uploaded_file.read().decode("utf-8")
            st.text_area("📄 Text preview", text_content[:3000], height=300)
        # ---------- Images ----------
        elif filename_lower.endswith((".png",".jpg","jpeg")):
            try:
                uploaded_file.seek(0)
                img = Image.open(uploaded_file)
                st.image(img, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
                st.text(f"Mode: {img.mode}, Size: {img.size[0]}x{img.size[1]}")
            except Exception as e:
                st.error(f"Failed to open image: {e}")
        # ---------- Chunking ----------
        chunks = []
        if text_content:
            chunks = chunk_text_chars(text_content, size=chunk_size, overlap=chunk_overlap)
            st.success(f"✅ Text chunked into {len(chunks)} chunks.")
            for i, c in enumerate(chunks[:3]):
                st.text_area(f"Chunk {i+1}", c[:1000], height=150)
        # ---------- Download ----------
        if chunks:
            if download_format == "Chunks (txt)":
                out_txt = "\n\n".join(chunks)
                st.download_button("⬇️ Download chunks (txt)", out_txt, file_name=f"{stats['name']}_chunks.txt")
            else:
                out_json = json.dumps(chunks, ensure_ascii=False, indent=2)
                st.download_button("⬇️ Download chunks (JSON)", out_json, file_name=f"{stats['name']}_chunks.json")
        # ---------- Chat ----------
        user_query = st.text_input("💬 Ask a question about this file:")
        if user_query and chunks:
            # simple retrieval
            scored = [(i, score_chunk_by_query(c, user_query)) for i, c in enumerate(chunks)]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_chunks = [chunks[i] for i, _ in scored[:max_chunks_for_context]]
            prompt = f"Context:\n{chr(10).join(top_chunks)}\n\nQuestion: {user_query}\nAnswer briefly."
            success, resp = ollama_query_http(OLLAMA_BASE_URL, HARDCODED_OLLAMA_MODEL, prompt)
            if not success:
                resp = local_fallback_summary(pdf_text=text_content, df=df, img=img, filename=stats['name'], query=user_query)
            st.markdown(f"**Response:**\n\n{resp}")

# ---------- CSV / XLSX Filter & Search ----------
elif view == "Filter & Search (CSV/XLSX)":
    uploaded_file = st.file_uploader("📂 Upload CSV / Excel file for filtering", type=["csv","xlsx","xls"], accept_multiple_files=False)
    if uploaded_file:
        try:
            filename_lower = uploaded_file.name.lower()
            if filename_lower.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            df = df_quick_clean(df)
            st.success(f"✅ Loaded file with {len(df):,} rows and {df.shape[1]} columns.")
            df_info_cards(df)
            # Filter UI
            selected_cols = st.multiselect("Select columns to filter:", df.columns.tolist(), default=df.columns.tolist())
            filters = {}
            for c in selected_cols:
                unique_vals = df[c].dropna().unique()
                if len(unique_vals) <= 30 and df[c].dtype != 'float64':
                    chosen = st.multiselect(f"Filter `{c}`:", unique_vals, default=unique_vals)
                    filters[c] = chosen
            filtered_df = df.copy()
            for c, vals in filters.items():
                filtered_df = filtered_df[filtered_df[c].isin(vals)]
            st.dataframe(filtered_df, height=300)
            # Search
            search_text = st.text_input("Search text across all columns")
            if search_text:
                mask = filtered_df.apply(lambda row: row.astype(str).str.contains(search_text, case=False, na=False).any(), axis=1)
                filtered_df = filtered_df[mask]
                st.dataframe(filtered_df, height=300)
            # Download filtered
            csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download filtered CSV", csv_bytes, file_name=f"{uploaded_file.name}_filtered.csv")
        except Exception as e:
            st.error(f"Failed to process file: {e}")

# ---------- About ----------
elif view == "About":
    st.markdown("""
    ### 📂 File Upload & Processing App
    - Developed by: **Isha Garg**
    - Features:
      - CSV / Excel / PDF / TXT / Image upload
      - Chunking and download
      - Filter & search for spreadsheets
      - Chatbot with context-aware retrieval
      - PDF OCR fallback
      - Styled cards and hover effects
    - Uses: Streamlit, pandas, pdfplumber, pytesseract, PIL, requests
    """)

import streamlit as st
import os
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import pytesseract
import datetime
import ollama
import json
import textwrap

# -------------------
# CONFIG
# -------------------
st.set_page_config(page_title="Policy Navigator", layout="wide")
DATASET_PATH = r"C:\Users\Ganesh\OneDrive\Desktop\DB"
HISTORY_FILE = os.path.join(DATASET_PATH, "chat_history.json")

# -------------------
# HELPERS
# -------------------
def save_chat_history():
    if "history" in st.session_state and st.session_state.history:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)

def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

@st.cache_data
def extract_text_from_pdf(pdf_bytes):
    text = ""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text")
        if text.strip():
            return text
        images = convert_from_bytes(pdf_bytes)
        for img in images:
            text += pytesseract.image_to_string(img, lang="eng")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
    return text

def query_ollama(messages, model="gemma:2b-instruct-q4_0"):
    try:
        response = ollama.chat(model=model, messages=messages)
        return response["message"]["content"]
    except Exception as e:
        return f"[Error contacting Ollama: {e}]"

# -------------------
# SESSION STATE
# -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "full_document_text" not in st.session_state:
    st.session_state.full_document_text = None
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None
if "history" not in st.session_state:
    st.session_state.history = load_chat_history()

# -------------------
# SIDEBAR
# -------------------
with st.sidebar:
    st.title("📜 Chat History")
    if st.button("New Chat"):
        if st.session_state.messages:
            first_question = st.session_state.messages[0]['content']
            title = (first_question[:30] + '...') if len(first_question) > 30 else first_question
            st.session_state.history.insert(0, {
                "date": datetime.date.today().strftime("%Y-%m-%d"),
                "title": title,
                "messages": st.session_state.messages
            })
            save_chat_history()
        st.session_state.messages = []
        st.session_state.full_document_text = None
        st.session_state.processed_file_name = None
        st.rerun()

    st.divider()
    for i, session in enumerate(st.session_state.history):
        if isinstance(session, dict) and 'title' in session:
            if st.button(f"{session.get('date', '')} - {session.get('title', 'Untitled')}", key=f"session_{i}"):
                st.session_state.messages = session.get('messages', [])
                st.rerun()

# -------------------
# MAIN CHAT
# -------------------
st.title("💬 Policy Navigator (Ollama Powered)")

if st.session_state.processed_file_name:
    st.info(f"Currently analyzing: **{st.session_state.processed_file_name}**")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.full_document_text:
        st.warning("Please upload a PDF document before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # --- CHUNKING CONTEXT ---
        full_text = st.session_state.full_document_text
        chunks = textwrap.wrap(full_text, width=1000, break_long_words=True, replace_whitespace=False)

        prompt_lower = prompt.lower()
        summary_keywords = ["summarize", "overview", "what is this", "about the policy", "explain the document"]
        stop_words = {'a', 'an', 'the', 'is', 'in', 'it', 'of', 'for', 'on', 'with', 'tell', 'me', 'about'}
        context_for_ai = ""

        if any(keyword in prompt_lower for keyword in summary_keywords):
            context_for_ai = "\n\n".join(chunks[:4])
        else:
            prompt_keywords = {word for word in prompt_lower.split() if word not in stop_words}
            relevant_chunks = []
            for chunk in chunks:
                if any(keyword in chunk.lower() for keyword in prompt_keywords):
                    relevant_chunks.append(chunk)
            if relevant_chunks:
                context_for_ai = "\n\n".join(relevant_chunks[:5])
            else:
                context_for_ai = "\n\n".join(chunks[:3])

        system_prompt = f"""
        You are a helpful assistant. Answer the user's question based ONLY on the following document context.
        If the answer is not found in the context, say "I cannot find the answer in the provided document."

        DOCUMENT CONTEXT:
        ---
        {context_for_ai}
        ---
        """

        messages_for_ollama = [{"role": "system", "content": system_prompt}]
        messages_for_ollama.extend(st.session_state.messages)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = query_ollama(messages_for_ollama)
                st.write(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

# -------------------
# FILE UPLOAD SECTION
# -------------------
st.divider()
st.subheader("📂 Upload Policy Files")

uploaded_file = st.file_uploader("Upload a PDF to start asking questions", type=["pdf"])

if uploaded_file and uploaded_file.name != st.session_state.processed_file_name:
    with st.spinner("Processing document... This may take a moment."):
        st.session_state.messages = []
        st.session_state.processed_file_name = uploaded_file.name

        pdf_bytes = uploaded_file.read()

        with st.expander("📁 File Processing Details", expanded=True):
            st.subheader("📄 PDF Preview")
            try:
                preview_images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1)
                st.image(preview_images[0], caption="Preview of the first page", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate a preview: {e}")

            text = extract_text_from_pdf(pdf_bytes)
            if text:
                st.subheader("🧩 Chunking & Storage")

                # ---- SAVE AS JSON (MODIFIED) ----
                chunks = textwrap.wrap(text, width=1000, break_long_words=True, replace_whitespace=False)
                json_data = {"file_name": uploaded_file.name, "chunks": chunks}
                save_json_path = os.path.join(DATASET_PATH, os.path.splitext(uploaded_file.name)[0] + ".json")
                os.makedirs(DATASET_PATH, exist_ok=True)
                with open(save_json_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)

                st.markdown(f"""
                - **Method**: The document is split into ~1000 character chunks.  
                - **Number of Chunks**: **{len(chunks)}**  
                - **Saved JSON**: `{save_json_path}`  
                """)

                # ---- SHOW SAMPLE TEXT (MODIFIED) ----
                st.subheader("🔍 Sample Extracted Text")
                st.write(text[:800] + "..." if len(text) > 800 else text)

                st.session_state.full_document_text = text
                st.success(f"✅ Document '{uploaded_file.name}' processed. You can now ask questions.")
            else:
                st.error("Failed to extract any text from the document.")
                st.session_state.full_document_text = None
                st.session_state.processed_file_name = None

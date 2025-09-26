import streamlit as st
import pdfplumber
import json
import subprocess
import math

st.title("📄 PDF Documentation Extractor + Chunking + Ollama")

uploaded_file = st.file_uploader("Upload your documentation PDF", type=["pdf"])

chunk_size = st.number_input("Enter chunk size (lines per chunk)", min_value=1, value=60, step=1)

ollama_model = "llama3.2:1b"

if "history" not in st.session_state:
    st.session_state.history = []

def query_ollama(model, prompt):
    try:
        result = subprocess.run(["ollama", "run", model], input=prompt.encode("utf-8"), capture_output=True)
        return result.stdout.decode("utf-8", errors="ignore").strip()
    except Exception as e:
        return f"❌ Error: {e}"

if uploaded_file is not None:
    full_text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"

    lines = [line.strip() for line in full_text.split("\n") if line.strip() != ""]

    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    st.subheader(f"🔹 Text Chunks (size = {chunk_size} lines)")

    for idx, chunk_lines in enumerate(chunks):
        st.markdown(f"### 📌 Chunk {idx+1}")
        chunk_text = "\n".join(chunk_lines)
        st.text_area(f"Chunk {idx+1} Text", chunk_text, height=200)

        chunk_json = json.dumps({"chunk_text": chunk_text}, indent=4)
        st.code(chunk_json, language="json")

        with open(f"output_chunk_{idx+1}.json", "w", encoding="utf-8") as f:
            f.write(chunk_json)

        user_prompt = st.text_area(f"💬 Ask Ollama about Chunk {idx+1}:", placeholder="Example: Summarize key insights, list all healthcare policies, etc.", key=f"prompt_{idx}")

        if st.button(f"Ask Ollama (Chunk {idx+1})", key=f"btn_{idx}"):
            if user_prompt.strip():
                final_prompt = f"Here is a documentation chunk:\n{chunk_json}\n\n{user_prompt}"
                response = query_ollama(ollama_model, final_prompt)
                st.session_state.history.append({"chunk": idx + 1, "question": user_prompt, "response": response})
                st.text_area(f"Ollama Response (Chunk {idx+1})", response, height=200)
            else:
                st.warning("⚠️ Please enter a prompt before asking Ollama.")

if st.session_state.history:
    st.subheader("📝 Interaction History")
    for i, entry in enumerate(st.session_state.history, 1):
        st.markdown(f"**{i}. Chunk {entry['chunk']}**")
        st.markdown(f"**Q:** {entry['question']}")
        st.markdown(f"**A:** {entry['response']}")
        st.markdown("---")

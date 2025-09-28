import threading
import requests
from flask import Flask, request, jsonify
import streamlit as st
import pandas as pd
from PIL import Image
import docx2txt
import fitz  # PyMuPDF
from transformers import pipeline
import wikipedia

# ------------------- Flask Backend -------------------
app = Flask(__name__)

# Load GPT-Neo model once (for chatbot)
chatbot = pipeline("text-generation", model="EleutherAI/gpt-neo-125m")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "")

    # First try Wikipedia
    try:
        wiki_answer = wikipedia.summary(query, sentences=2, auto_suggest=True, redirect=True)
        return jsonify({"answer": wiki_answer})
    except wikipedia.exceptions.DisambiguationError as e:
        return jsonify({"answer": f"Your query is ambiguous. Try one of these: {e.options[:5]}"})
    except wikipedia.exceptions.PageError:
        pass
    except Exception:
        pass

    # Fallback to AI model
    res = chatbot(query, max_length=50, do_sample=True, temperature=0.7)
    return jsonify({"answer": res[0]["generated_text"]})


def run_flask():
    app.run(port=5000, debug=False, use_reloader=False)


# ------------------- Streamlit Frontend -------------------
def run_streamlit():
    st.set_page_config(page_title="AI Chatbot + File Uploader", layout="wide")
    st.title("🤖 AI Chatbot with File Uploader")
    st.markdown("Chat with AI (via Flask backend) or upload files for preview")

    # Sidebar navigation
    app_mode = st.sidebar.selectbox("Choose Mode", ["Chatbot", "File Uploader"])

    # ------------------- Chatbot Mode -------------------
    if app_mode == "Chatbot":
        st.subheader("💬 Chat with AI + Wikipedia")
        user_input = st.text_input("You:", placeholder="Ask me anything...")
        if user_input:
            try:
                res = requests.post("http://localhost:5000/chat", json={"query": user_input})
                if res.status_code == 200:
                    st.success("Answer:")
                    st.write(res.json()["answer"])
                else:
                    st.error("Backend error. Try again.")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")

    # ------------------- File Uploader Mode -------------------
    elif app_mode == "File Uploader":
        st.subheader("📂 Multi-File Uploader")
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=["csv", "pdf", "docx", "txt", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
            show_meta = st.checkbox("Show file metadata")

            for file in uploaded_files:
                st.divider()
                file_name = file.name
                file_type = file.type
                file_size_kb = round(len(file.getvalue()) / 1024, 2)
                file_ext = file_name.split(".")[-1].lower()

                # 🔍 Metadata
                if show_meta:
                    st.markdown(f"**File Name:** {file_name}")
                    st.markdown(f"**File Type:** {file_type}")
                    st.markdown(f"**File Size:** {file_size_kb} KB")

                # 🖼️ Preview
                try:
                    if file_ext == "csv":
                        df = pd.read_csv(file)
                        st.dataframe(df)

                    elif file_ext == "txt":
                        text = file.getvalue().decode("utf-8", errors="ignore")
                        st.text_area("TXT Preview", text, height=200)

                    elif file_ext == "docx":
                        text = docx2txt.process(file)
                        st.text_area("DOCX Preview", text, height=200)

                    elif file_ext == "pdf":
                        pdf = fitz.open(stream=file.read(), filetype="pdf")
                        text = ""
                        for page in pdf:
                            text += page.get_text()
                        st.text_area("PDF Preview", text, height=200)

                    elif file_ext in ["png", "jpg", "jpeg"]:
                        img = Image.open(file)
                        st.image(img, caption=file_name, use_column_width=True)

                    else:
                        st.warning("Unsupported file format.")
                except Exception as e:
                    st.error(f"Error processing {file_name}: {e}")
        else:
            st.info("Please upload one or more files to begin.")


# ------------------- Main Entry -------------------
if __name__ == "__main__":
    # Start Flask in background thread
    threading.Thread(target=run_flask, daemon=True).start()

    # Run Streamlit (blocks)
    run_streamlit()

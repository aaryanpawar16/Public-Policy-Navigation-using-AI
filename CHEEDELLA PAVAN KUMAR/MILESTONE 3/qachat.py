from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
import docx
from gtts import gTTS
import base64
from PIL import Image
import io
import requests
import openai
import speech_recognition as sr
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import tempfile

# -------------------- API Keys --------------------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

# -------------------- Gemini Chat Setup --------------------
model_text = genai.GenerativeModel("gemini-2.5-pro")
chat = model_text.start_chat(history=[])

def get_gemini_response(prompt):
    response = chat.send_message(prompt, stream=True)
    return response

# -------------------- File Processing --------------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    return text

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8")

# -------------------- Text-to-Speech --------------------
def text_to_speech(text, filename=None):
    if not filename:
        filename = f"{tempfile.mktemp()}.mp3"
    tts = gTTS(text)
    tts.save(filename)
    return filename

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
    md = f"""
    <audio autoplay controls>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

# -------------------- Voice-to-Text --------------------
def voice_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now")
        audio_data = recognizer.listen(source, phrase_time_limit=10)
        try:
            text = recognizer.recognize_google(audio_data)
            st.success(f" You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error(" Could not understand audio")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
    return ""

# -------------------- Image Generation --------------------
def generate_image(prompt):
    # Stability AI
    if STABILITY_API_KEY:
        try:
            url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
            headers = {"Authorization": f"Bearer {STABILITY_API_KEY}"}
            json_data = {
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7,
                "height": 1024,
                "width": 1024,
                "samples": 1,
                "steps": 30
            }
            response = requests.post(url, headers=headers, json=json_data)
            response.raise_for_status()
            data = response.json()
            image_base64 = data['artifacts'][0]['base64']
            image_bytes = base64.b64decode(image_base64)
            return Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            st.warning(f"Stability AI failed, trying OpenAI: {e}")

    # OpenAI fallback
    try:
        response = openai.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024"
        )
        image_bytes = base64.b64decode(response.data[0].b64_json)
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        st.warning(f"OpenAI image generation failed, showing placeholder: {e}")
        return Image.new("RGB", (512, 512), color=(200, 200, 200))

# -------------------- Chunking --------------------
def create_chunks(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.create_documents([text])
    return [chunk.page_content for chunk in chunks]

def create_audio_chunks(chunks, prefix="chunk"):
    audio_files = []
    for i, c in enumerate(chunks):
        filename = f"{prefix}_{i+1}.mp3"
        text_to_speech(c, filename)
        audio_files.append(filename)
    return audio_files

# -------------------- Streamlit App --------------------
st.set_page_config(page_title="Ollama Chatbot", layout="wide")
st.header("Ollama Chatbot")

# Sidebar navigation
menu = st.sidebar.selectbox("Navigate", ["Chatbot", "Document Q&A / Summarization", "Text to Image", "Chunks", "Chat History"])

# Session State
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "doc_content" not in st.session_state:
    st.session_state["doc_content"] = ""
if "chunks" not in st.session_state:
    st.session_state["chunks"] = {"chat": [], "document": [], "image_prompt": []}
if "audio_chunks" not in st.session_state:
    st.session_state["audio_chunks"] = {"chat": [], "document": [], "image_prompt": []}

# -------------------- Chatbot Section --------------------
if menu == "Chatbot":
    st.subheader("Chat with Ollama")
    output_mode = st.radio("Choose output mode:", ["Text", "Voice"], horizontal=True)
    input_type = st.radio("Input type:", ["Type", "Speak"], horizontal=True)
    chunk_size = st.number_input("Chat chunk size:", min_value=20, max_value=1024, value=100)
    chunk_overlap = st.number_input("Chat chunk overlap:", min_value=0, max_value=chunk_size-1, value=20)

    user_input = ""
    if input_type == "Type":
        user_input = st.text_input("Input your question:", key="input_chat")
    else:
        if st.button("Speak your question"):
            user_input = voice_to_text()

    if st.button("Ask Ollama") and user_input:
        response = get_gemini_response(user_input)
        collected_text = ""
        for chunk in response:
            st.write(chunk.text)
            collected_text += chunk.text
            st.session_state["chat_history"].append(("Bot", chunk.text))
        st.session_state["chat_history"].append(("You", user_input))

        # Generate chat chunks
        st.session_state["chunks"]["chat"] = create_chunks(collected_text, chunk_size, chunk_overlap)
        st.session_state["audio_chunks"]["chat"] = create_audio_chunks(st.session_state["chunks"]["chat"], prefix="chat_chunk")

        if output_mode == "Voice":
            for audio_file in st.session_state["audio_chunks"]["chat"]:
                autoplay_audio(audio_file)

    if st.session_state["chunks"]["chat"]:
        st.subheader("Chat Chunks")
        for i, c in enumerate(st.session_state["chunks"]["chat"]):
            st.code(f"Chunk {i+1}:\n{c}")
        st.download_button("Download Chat Chunks", json.dumps(st.session_state["chunks"]["chat"], indent=2), "chat_chunks.json")
        st.download_button("Download Chat Audio Chunks (ZIP not implemented here)", "Audio files are saved in temp folder.", "chat_audio_info.txt")

# -------------------- Document Q&A / Summarization --------------------
elif menu == "Document Q&A / Summarization":
    st.subheader("Upload a Document for Q&A / Summarization")
    uploaded_file = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
    chunk_size = st.number_input("Document chunk size:", min_value=50, max_value=2048, value=250)
    chunk_overlap = st.number_input("Document chunk overlap:", min_value=0, max_value=chunk_size-1, value=50)

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text = extract_text_from_txt(uploaded_file)
        else:
            text = ""

        st.session_state["doc_content"] = text
        st.subheader("Document Content (Preview)")
        st.write(text[:1000] + "...")

        if st.button("Summarize Document"):
            summary_prompt = f"Summarize this document:\n\n{text}"
            response = get_gemini_response(summary_prompt)
            collected_text = ""
            for chunk in response:
                st.write(chunk.text)
                collected_text += chunk.text
            # Save document chunks
            st.session_state["chunks"]["document"] = create_chunks(collected_text, chunk_size, chunk_overlap)
            st.session_state["audio_chunks"]["document"] = create_audio_chunks(st.session_state["chunks"]["document"], prefix="doc_chunk")

        if st.session_state["chunks"]["document"]:
            st.subheader("Document Chunks")
            for i, c in enumerate(st.session_state["chunks"]["document"]):
                st.code(f"Chunk {i+1}:\n{c}")
            st.download_button("Download Document Chunks", json.dumps(st.session_state["chunks"]["document"], indent=2), "document_chunks.json")
            st.download_button("Download Document Audio Chunks (ZIP not implemented)", "Audio files are saved in temp folder.", "doc_audio_info.txt")

# -------------------- Text to Image Section --------------------
elif menu == "Text to Image":
    st.subheader(" Generate Image from Text")
    image_prompt = st.text_input("Enter a description to generate an image:", key="image_prompt_nav")
    chunk_size = st.number_input("Image prompt chunk size:", min_value=20, max_value=1024, value=100)
    chunk_overlap = st.number_input("Image prompt chunk overlap:", min_value=0, max_value=chunk_size-1, value=20)

    if st.button("Generate Image") and image_prompt:
        img = generate_image(image_prompt)
        st.image(img, caption=image_prompt)
        # Save prompt chunks
        st.session_state["chunks"]["image_prompt"] = create_chunks(image_prompt, chunk_size, chunk_overlap)
        st.session_state["audio_chunks"]["image_prompt"] = create_audio_chunks(st.session_state["chunks"]["image_prompt"], prefix="img_prompt_chunk")

        if st.session_state["chunks"]["image_prompt"]:
            st.subheader("Image Prompt Chunks")
            for i, c in enumerate(st.session_state["chunks"]["image_prompt"]):
                st.code(f"Chunk {i+1}:\n{c}")
            st.download_button("Download Image Prompt Chunks", json.dumps(st.session_state["chunks"]["image_prompt"], indent=2), "image_prompt_chunks.json")
            st.download_button("Download Image Prompt Audio Chunks (ZIP not implemented)", "Audio files are saved in temp folder.", "img_audio_info.txt")

# -------------------- Chunks Page --------------------
elif menu == "Chunks":
    st.subheader("All Chunks")
    for section, chunk_list in st.session_state["chunks"].items():
        st.markdown(f"### {section.capitalize()} Chunks ({len(chunk_list)})")
        for i, c in enumerate(chunk_list):
            st.code(f"Chunk {i+1}:\n{c}")

# -------------------- Chat History --------------------
elif menu == "Chat History":
    st.subheader("Chat History")
    for role, text in st.session_state["chat_history"]:
        st.write(f"{role}: {text}")

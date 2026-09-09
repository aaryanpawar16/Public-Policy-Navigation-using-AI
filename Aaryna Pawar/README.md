# 📄 Public Policy Navigation using AI — PDF OCR + Ollama Chat

A Streamlit application that extracts text from scanned/image-based PDFs using **OCR (Tesseract)**, splits the extracted text into searchable chunks, retrieves the most relevant chunks for a user's question (via **TF-IDF cosine similarity** or a naive word-overlap fallback), and generates answers using a **locally running Ollama LLM**.

This tool is designed for navigating dense documents — such as public policy papers, government notices, or scanned reports — by letting users "chat" with the PDF content.

---

## ✨ Features

- **OCR Text Extraction** — Converts each PDF page to an image (via Poppler) and extracts text using Tesseract OCR
- **Text Chunking** — Splits extracted page text into overlapping chunks for better retrieval
- **Smart Retrieval** — Uses scikit-learn's TF-IDF + cosine similarity to find the most relevant chunks for a query (falls back to simple word-overlap scoring if scikit-learn isn't installed)
- **Local LLM Chat** — Sends the retrieved context + question to a locally running **Ollama** model and returns an answer
- **Interactive UI** — Upload a PDF, preview extracted pages/chunks, adjust OCR/chunking settings, and chat — all from the browser
- **Dependency Health Check** — Automatically checks whether Tesseract and Poppler are found and accessible

---

## 🧠 How It Works

1. **Upload** a PDF file through the Streamlit interface
2. **OCR** — Each page is rendered as an image (via `pdf2image`/Poppler) and passed through `pytesseract` to extract text
3. **Chunking** — Extracted page text is split into overlapping chunks (configurable size/overlap) to preserve context across boundaries
4. **Retrieval** — When a question is asked, the app scores all chunks against the query (TF-IDF cosine similarity, or naive term-overlap if scikit-learn is unavailable) and picks the top-K most relevant chunks
5. **Generation** — The question + retrieved context are sent as a prompt to a local **Ollama** model (`/api/generate`), and the response is displayed in a chat-style interface

---

## 🛠️ Tech Stack

| Component | Purpose |
|---|---|
| [Streamlit](https://streamlit.io/) | Web UI |
| [pytesseract](https://pypi.org/project/pytesseract/) | OCR engine wrapper |
| [pdf2image](https://pypi.org/project/pdf2image/) | Converts PDF pages to images |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | Underlying OCR engine (external binary) |
| [Poppler](https://poppler.freedesktop.org/) | PDF rendering utility (external binary, `pdftoppm`) |
| [scikit-learn](https://scikit-learn.org/) *(optional)* | TF-IDF vectorization & cosine similarity for retrieval |
| [Ollama](https://ollama.com/) | Local LLM inference (e.g., `llama2`, `mistral`, `qwen`) |
| [Pillow (PIL)](https://pypi.org/project/Pillow/) | Image handling |
| `requests` | HTTP calls to the local Ollama API |

---

## 📦 Prerequisites

Before running the app, make sure the following are installed **on your system** (not just via `pip`):

1. **Tesseract OCR** — [Download/install instructions](https://github.com/UB-Mannheim/tesseract/wiki) (Windows) or `sudo apt install tesseract-ocr` (Linux) / `brew install tesseract` (macOS)
2. **Poppler** — [Windows builds](https://github.com/oschwartz10612/poppler-windows/releases) or `sudo apt install poppler-utils` (Linux) / `brew install poppler` (macOS)
3. **Ollama** — [Install Ollama](https://ollama.com/download), then pull a model:
   ```bash
   ollama pull llama2
   ollama run llama2
   ```
   (Ollama must be running locally on `http://localhost:11434` for the chat feature to work.)

---

## ⚙️ Setup

1. **Clone the repository**
   ```bash
   git clone [https://github.com/aaryanpawar16/Public-Policy-Navigation-using-AI/tree/main/Aaryna%20Pawar](https://github.com/aaryanpawar16/Public-Policy-Navigation-using-AI/tree/main/Aaryna%20Pawar)
   ```

2. **Install Python dependencies**
   ```bash
   pip install streamlit pytesseract pdf2image pillow requests scikit-learn
   ```

3. **Update hardcoded paths**

   The script currently points to Windows-specific paths for Tesseract and Poppler:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r"F:\tesseract ocr\tesseract.EXE"
   POPPLER_PATH = r"F:\poppler-25.07.0\Library\bin"
   ```
   Update these to match your local installation paths (or remove them if the binaries are already in your system `PATH`).

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. Open the app in your browser (Streamlit will show a local URL, typically `http://localhost:8501`).

---

## 🚀 Usage

1. Upload a PDF file using the file uploader
2. Click **"Start OCR and Chunking"** to extract and process the text
3. Optionally preview extracted pages and text chunks
4. Type a question about the document in the chat box and click **Submit**
5. The app retrieves relevant chunks and generates an answer using your local Ollama model
6. Conversation history is shown below the input box

### Sidebar Settings
| Setting | Description |
|---|---|
| **DPI** | Resolution used when converting PDF pages to images (higher = better OCR accuracy, slower) |
| **Chunk size** | Number of characters per text chunk |
| **Overlap** | Number of overlapping characters between consecutive chunks |
| **Top K chunks** | Number of most relevant chunks retrieved per question |

---

## ⚠️ Notes & Limitations

- Requires **Ollama running locally** — the app will return an error message in the chat if it can't reach `http://localhost:11434`
- OCR accuracy depends on scan quality, DPI setting, and document formatting
- The retrieval step is a simple TF-IDF/keyword-based method, not a full vector database — best suited for small to medium-sized PDFs
- Paths for Tesseract/Poppler are currently hardcoded for a Windows environment and should be adjusted for your OS

---

## 📌 Possible Improvements

- Replace hardcoded paths with environment variables or a config file
- Add support for other local LLM backends
- Use a proper vector store (e.g., FAISS, Chroma) for larger document retrieval
- Add multi-PDF support and persistent chat history
- Display OCR confidence scores per page

---

## 👤 Author

**Aaryan Pawar**
AI Intern 

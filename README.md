# 📄 Public Policy Navigation Using AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📘 Overview
This project is part of our **Infosys Springboard Virtual Internship – AI Track**.  
It focuses on using **AI-powered document understanding** to simplify navigation of **public policy documents**.  

The application extracts text from PDFs, splits it into smaller chunks, and enables interactive question-answering with **Ollama AI models**.

---

## 🛠️ Features
- 📂 Upload PDF documentation  
- ✂️ Automatic text extraction and cleaning  
- 🔎 Chunking large documents into manageable pieces  
- 🗂️ Convert chunks into JSON format  
- 🤖 Query **Ollama AI** on specific document chunks  
- 📝 Maintain interaction history for review  

---

## 🚀 Tech Stack
- **Python 3**  
- **Streamlit** – Web app framework  
- **pdfplumber** – PDF text extraction  
- **Ollama CLI** – AI model integration  
- **JSON** – Data formatting  

---



## 📂 Project Structure <br>
📦 Public-Policy-Navigation-AI <br>
┣ 📜 app.py # Streamlit application <br>
┣ 📜 README.md # Documentation 


---

## ▶️ How to Run
1. Clone the repository:
   
   git clone https://github.com/aaryanpawar16/Public-Policy-Navigation-using-AI.git
   cd Public-Policy-Navigation-using-AI

Run the app:
streamlit run app.py

2. Install dependencies:
   
   pip install streamlit pdfplumber

3.Ensure Ollama is installed and running:

   ollama run llama3

Run the app:

streamlit run app.py
💡 Use Cases
📜 Simplifying government policy documents
🎓 Academic research assistance
⚖️ Legal document exploration
🏢 Corporate compliance analysis
🔮 Future Improvements
🌐 Multi-language support
📊 Advanced search & filtering
🧠 Improved chunking using embeddings
🔗 Integration with vector databases (FAISS / Chroma)
🎤 Voice-based query interface
🤝 Contributing

Contributions are welcome!

Fork the repository
Create your feature branch
Commit your changes
Push to the branch
Open a Pull Request
📜 License

This project is licensed under the MIT License – feel free to use and modify.

## 🔖 Tags
`#AI` `#Internship` `#InfosysSpringboard` `#Python`

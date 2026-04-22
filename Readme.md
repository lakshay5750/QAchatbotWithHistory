# 📄 Conversational RAG with PDF + Chat History

A Streamlit-based Conversational AI app and store the history that allows users to upload PDF files and chat with them using a Retrieval-Augmented Generation (RAG) pipeline powered by LangChain, Groq, and ChromaDB.

---

## 🚀 Features

- 📚 Upload multiple PDF files
- 🤖 Chat with PDFs using LLM (Llama-3 via Groq)
- 🧠 Context-aware responses using chat history
- 🔍 Semantic search using HuggingFace embeddings
- 🗂️ PDF chunking for better retrieval
- 💬 Session-based memory support
- ⚡ Fast vector search using ChromaDB

---

## 🏗️ Architecture

 PDF Upload → Text Extraction → Chunking → Embeddings → Vector DB (Chroma)
                                              ↓
User Query → Context Retrieval → LLM (Groq) → Final Answer + Chat History

---

## 🛠️ Tech Stack

- Streamlit (Frontend)
- LangChain (RAG Framework)
- Groq API (Llama-3.3-70B)
- ChromaDB (Vector Store)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- PyPDF (PDF processing)
- python-dotenv (Environment variables)

---

## 📦 Installation

### 1. Clone Repository
- git clone https://github.com/lakshay5750/QAchatbotWithHistory.git
- cd QAchatbotWithHistory

---

### 2. Create Virtual Environment
- python -m venv venv
- source venv/bin/activate   # Mac/Linux
- venv\Scripts\activate      # Windows

---

### 3. Install Dependencies
pip install -r requirements.txt

---

## 🔑 Environment Variables

Create a `.env` file:


- HF_TOKEN=your_huggingface_token

---

## ▶️ Run Application

streamlit run app.py

---

## 📁 Project Structure

```
project/
│── app.py
│── requirements.txt
│── .env
│── .gitignore
│── README.md
```

---

## 📌 Requirements

- streamlit
- langchain
- langchain-core
- langchain-community
- langchain-classic
- langchain-groq
- langchain-huggingface
- langchain-text-splitters
- chromadb
- python-dotenv
- sentence-transformers
- pypdf
- ipykernel

---

## ⚠️ Notes

- PDF embeddings are created on every run
- Large PDFs may take time to process
- Do NOT upload `.env` to GitHub
- ChromaDB is currently in-memory (not persistent)

---

## 🔒 Security
```
Files ignored in Git:
.env
.lakshay
```

---

## 🚀 Future Improvements

- Persistent vector storage (Chroma/Faiss disk mode)
- Faster retrieval optimization
- PDF page-level citations
- Streaming responses
- Multi-user authentication

---

## 👨‍💻 Author

Lakshya Varshney

---

## ⭐ If you like this project
Give it a star ⭐ and share it 🚀

[Live App](https://appchatbotwithhistory-logapprbb4ceddpaecjks6g.streamlit.app/)

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/1ca542d7-4005-4872-acbd-9c08545d29a6" />


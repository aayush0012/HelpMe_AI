# HelpMe AI

HelpMe AI is an AI-powered academic assistant built using a Retrieval-Augmented Generation (RAG) architecture that enables users to upload PDF documents and interact with them through natural language queries. The platform performs intelligent document ingestion, semantic chunking, vector embedding generation, semantic retrieval, and context-aware answer generation from uploaded academic material.

The system is designed to simplify navigation and understanding of large academic PDFs through conversational interaction and AI-assisted knowledge retrieval.

---

# Features

* Upload and process academic PDF documents
* Semantic search over uploaded notes
* Context-aware question answering
* Retrieval-Augmented Generation (RAG) pipeline
* Conversational AI interface
* Real-time PDF querying
* Vector-based semantic retrieval
* Interactive frontend with chat interface
* FastAPI backend APIs
* Persistent vector database using ChromaDB

---

# System Architecture

PDF Upload
→ Document Parsing
→ Semantic Chunking
→ Embedding Generation
→ ChromaDB Vector Storage
→ Semantic Retrieval
→ LLM Response Generation

---

# Tech Stack

## Frontend

* React.js
* Axios

## Backend

* FastAPI
* Python

## AI / ML

* LangChain
* Ollama
* HuggingFace Embeddings

## Vector Database

* ChromaDB

## PDF Processing

* Unstructured

---

# Project Structure

```bash
HelpMe-AI/
│
├── frontend/
│   ├── src/
│   └── package.json
│
├── uploads/
│
├── chroma_db/
│
├── document_ingestion.py
├── retrieval.py
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/aayush0012/HelpMe-AI.git

cd HelpMe-AI
```

---

# Backend Setup

## Create Virtual Environment

```bash
python -m venv .venv
```

## Activate Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Install Ollama

Download Ollama from:

https://ollama.com/

Pull model:

```bash
ollama pull llama3.2
```

Start Ollama:

```bash
ollama serve
```

---

# Frontend Setup

Move into frontend folder:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Run frontend:

```bash
npm run dev
```

---

# Run Backend

Inside root directory:

```bash
uvicorn main:app --reload
```

---

# API Endpoints

## Upload PDF

```http
POST /upload
```

Uploads and processes academic PDFs.

---

## Chat With Notes

```http
POST /chat
```

Accepts user query and returns context-aware AI-generated response.

---

# Workflow

1. User uploads PDF
2. PDF is parsed and chunked
3. Embeddings are generated
4. Chunks stored in ChromaDB
5. User asks question
6. Relevant chunks retrieved semantically
7. LLM generates contextual response

---

# Future Improvements

* Multi-PDF support
* Conversational memory
* Authentication system
* Cloud deployment
* Citation-based answers
* Hybrid search
* Streaming responses
* OCR optimization

---

# Author

Aayush Bhatt

GitHub:
https://github.com/aayush0012

LinkedIn:
https://www.linkedin.com/in/aayush-bhatt-3657b1314/

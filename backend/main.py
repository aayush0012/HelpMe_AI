
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import shutil
import os

from backend.document_ingestion import (
    partition_document,
    chunk_document,
    process_chunks,
    create_vectorstore
)

from backend.retrieval import load_retriever

from langchain_ollama import ChatOllama


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

llm = ChatOllama(
    model="llama3.2:latest"
)


@app.get("/")
def home():

    return {
        "message": "RAG Backend Running"
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if os.path.exists("./chroma_db"):

        shutil.rmtree(
            "./chroma_db",
            ignore_errors=True
        )

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    elements = partition_document(file_path)

    chunks = chunk_document(elements)

    processed_documents = process_chunks(chunks)

    create_vectorstore(processed_documents)

    return {
        "message": "PDF processed successfully"
    }


@app.post("/chat")
async def chat(question: str):

    retriever = load_retriever()

    results = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    prompt = f"""
You are an academic assistant.

Answer the question strictly using the provided context.

Rules:
- Do not use outside knowledge
- If answer is not present, say:
  "Information not found in notes."
- Keep answers clear and accurate
- Preserve technical terminology
- If context contains partially relevant information,
  infer carefully.

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


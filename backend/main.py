import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import shutil
import traceback
import re
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from document_ingestion import (
    partition_document,
    chunk_document,
    process_chunks,
    create_vectorstore,
)
from hybrid_retrieval import hybrid_retrieve
from agent import StudyAgent
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

base_dir = os.path.dirname(os.path.abspath(__file__))
upload_folder = os.path.join(base_dir, "uploads")
persist_dir = os.path.join(base_dir, "chroma_db")
frontend_dist_path = os.path.abspath(os.path.join(base_dir, "../frontend/dist"))

os.makedirs(upload_folder, exist_ok=True)

def get_session_paths(session_id, create=True):
    if not session_id or not re.match(r"^[a-zA-Z0-9_\-]+$", session_id):
        return upload_folder, persist_dir
    
    session_upload = os.path.join(upload_folder, session_id)
    session_persist = os.path.join(persist_dir, session_id)
    if create:
        os.makedirs(session_upload, exist_ok=True)
        os.makedirs(session_persist, exist_ok=True)
    return session_upload, session_persist

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "https://help-me-two-sigma.vercel.app",
]
cors_env = os.getenv("CORS_ALLOWED_ORIGINS")
if cors_env:
    cleaned_cors = cors_env.strip()
    if cleaned_cors == "*":
        allowed_origins = ["*"]
    else:
        allowed_origins = []
        parts = cors_env.split(",")
        for i in range(len(parts)):
            item = parts[i].strip()
            if item:
                allowed_origins.append(item)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY environment variable is not set. Please configure it in your settings."
        )
    primary_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
    )
    fallback_llm1 = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
    )
    fallback_llm2 = ChatGroq(
        model="mixtral-8x7b-32768",
        api_key=api_key,
    )
    return primary_llm.with_fallbacks([fallback_llm1, fallback_llm2])

def get_embeddings():
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        raise HTTPException(
            status_code=500,
            detail="HUGGINGFACEHUB_API_TOKEN or HF_TOKEN environment variable is not set. Please configure it in your settings."
        )
    embeds = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=hf_token
    )
    return embeds

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def home():
    index_file = os.path.join(frontend_dist_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "RAG Backend Running"}

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Query(None)
):
    try:
        print("Upload started (session: " + str(session_id) + ")")
        filename = file.filename
        lower_name = filename.lower()
        if not lower_name.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        safe_filename = os.path.basename(filename)
        session_upload_folder, session_persist_dir = get_session_paths(session_id)

        if os.path.exists(session_persist_dir):
            try:
                import chromadb
                client = chromadb.PersistentClient(path=session_persist_dir)
                collections = client.list_collections()
                for i in range(len(collections)):
                    col = collections[i]
                    client.delete_collection(col.name)
            except Exception as e:
                print(e)

        print("Saving file")
        file_path = os.path.join(session_upload_folder, safe_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("Partitioning PDF")
        elements = partition_document(file_path)

        print("Chunking elements")
        chunks = chunk_document(elements)

        print("Processing chunks")
        processed_documents = process_chunks(chunks, source_name=safe_filename)

        if len(processed_documents) == 0:
            raise HTTPException(
                status_code=422,
                detail="No usable content could be extracted from this PDF",
            )

        print("Creating vectorstore")
        create_vectorstore(processed_documents, persist_directory=session_persist_dir)

        print("Done")
        res = {
            "message": "PDF processed successfully",
            "chunks_indexed": len(processed_documents)
        }
        return res

    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to process PDF")

@app.post("/chat")
async def chat(
    question: str = Query(...),
    session_id: str = Query(None)
):
    question = question.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    session_upload_folder, session_persist_dir = get_session_paths(session_id, create=False)

    if not os.path.exists(session_persist_dir) or not os.listdir(session_persist_dir):
        raise HTTPException(
            status_code=400,
            detail="No document has been ingested yet. Upload a PDF first.",
        )

    vectorstore = Chroma(
        persist_directory=session_persist_dir,
        embedding_function=get_embeddings(),
    )

    agent = StudyAgent(vectorstore=vectorstore, llm=get_llm())
    result = agent.invoke(question)

    res = {
        "answer": result["answer"],
        "sources": result["sources"],
        "steps": result["steps"]
    }
    return res

if os.path.exists(frontend_dist_path):
    assets_path = os.path.join(frontend_dist_path, "assets")
    app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    @app.get("/{catchall:path}")
    async def serve_frontend(catchall: str):
        if catchall.startswith("upload") or catchall.startswith("chat"):
            raise HTTPException(status_code=404, detail="Not Found")

        file_path = os.path.join(frontend_dist_path, catchall)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)

        index_file = os.path.join(frontend_dist_path, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Frontend build index.html not found")
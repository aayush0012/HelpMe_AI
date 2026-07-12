import os
import hashlib
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
import fitz
import base64
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
EMBED_BATCH_SIZE = 100

def run_cloud_ocr(file_path: str):
    print("Opening PDF with PyMuPDF for cloud transcription...")
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"Failed to open PDF with PyMuPDF: {e}")
        return []

    ocr_elements = []
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Warning: GROQ_API_KEY is not set. Cloud OCR will be skipped.")
        return []

    try:
        vision_llm = ChatGroq(
            model="llama-3.2-11b-vision-preview",
            api_key=api_key,
            temperature=0
        )
    except Exception as e:
        print(f"Failed to initialize ChatGroq Vision: {e}")
        return []

    for page_num in range(len(doc)):
        print(f"  Running OCR transcription on page {page_num + 1}/{len(doc)}...")
        try:
            page = doc[page_num]
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")

            base64_image = base64.b64encode(img_bytes).decode("utf-8")

            message = HumanMessage(
                content=[
                    {
                        "type": "text", 
                        "text": "Transcribe all text, numbers, and structured table data from this page image exactly as they appear. Do not summarize, format, or add any commentary. Output only the transcribed text."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            )

            response = vision_llm.invoke([message])
            transcribed_text = response.content.strip()

            if transcribed_text:
                doc_element = Document(
                    page_content=transcribed_text,
                    metadata={
                        "source": os.path.basename(file_path),
                        "page": page_num
                    }
                )
                ocr_elements.append(doc_element)
        except Exception as e:
            print(f"  Error transcribing page {page_num + 1}: {e}")

    return ocr_elements

def partition_document(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at: {file_path}")

    print(f"Loading '{file_path}' with PyPDFLoader...")
    start = time.time()
    try:
        loader = PyPDFLoader(file_path)
        elements = loader.load()
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF '{file_path}': {e}") from e

    total_chars = 0
    for doc in elements:
        total_chars += len(doc.page_content.strip())
    
    print(f"Standard load complete. Total characters: {total_chars}")

    if total_chars < 150:
        print("Standard PDF loader extracted minimal text. Falling back to Cloud Vision OCR...")
        ocr_elements = run_cloud_ocr(file_path)
        if ocr_elements:
            elements = ocr_elements
            print(f"Cloud OCR complete. Pages transcribed: {len(elements)}")
        else:
            print("Cloud OCR returned no pages. Using standard loader output.")

    elapsed = time.time() - start
    print(f"Ingestion load completed in {elapsed:.1f}s")
    return elements

def chunk_document(elements):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(elements)
    return chunks

def process_chunks(chunks, source_name):
    documents = []
    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", 0) + 1
        
        if len(chunk.page_content.strip()) < 10:
            print(f"Skipping empty chunk {i}")
            continue

        doc = Document(
            page_content=chunk.page_content,
            metadata={
                "source": source_name,
                "pages": str(page),
                "chunk_index": i,
            },
        )
        documents.append(doc)

    print("Processed docs:", len(documents))
    return documents

def make_doc_id(doc: Document) -> str:
    key = f"{doc.metadata.get('source')}::chunk_{doc.metadata.get('chunk_index')}::{doc.page_content}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def create_vectorstore(documents, persist_directory=None):
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HUGGINGFACEHUB_API_TOKEN is not set. Cloud embedding calls might fail.")

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=hf_token
    )

    if persist_directory is None:
        persist_directory = PERSIST_DIR

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    ids = []
    for doc in documents:
        ids.append(make_doc_id(doc))

    print(f"Inserting {len(documents)} documents in batches of {EMBED_BATCH_SIZE}...")

    for start in range(0, len(documents), EMBED_BATCH_SIZE):
        end = start + EMBED_BATCH_SIZE
        batch_docs = documents[start:end]
        batch_ids = ids[start:end]

        vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
        print(f"  Inserted {min(end, len(documents))}/{len(documents)}")

    return vectorstore

if __name__ == "__main__":
    file_path = os.path.join("docs", "rag.pdf")
    source_name = os.path.basename(file_path)

    try:
        print("Loading PDF...")
        elements = partition_document(file_path)

        print("Creating chunks...")
        chunks = chunk_document(elements)

        print("Processing chunks...")
        processed_documents = process_chunks(chunks, source_name)

        if not processed_documents:
            print("No documents produced from this PDF — nothing to ingest.")
        else:
            print("Generating embeddings and storing vectors...")
            vectorstore = create_vectorstore(processed_documents)
            print("Ingestion completed")

    except FileNotFoundError as e:
        print(f"File error: {e}")
    except RuntimeError as e:
        print(f"Processing error: {e}")
    except Exception as e:
        print(f"Unexpected error during ingestion: {e}")
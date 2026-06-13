
import os
from dotenv import load_dotenv

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def partition_document(file_path: str):

    elements = partition_pdf(
        filename=file_path,
        strategy="fast",
        infer_table_structure=True
    )

    return elements


def chunk_document(elements):

    chunks = chunk_by_title(
        elements=elements,
        max_characters=1200,
        combine_text_under_n_chars=300
    )

    return chunks


def separate_content(chunk):

    data = {
        "text": chunk.text,
        "tables": []
    }

    if hasattr(chunk.metadata, "orig_elements"):

        for item in chunk.metadata.orig_elements:

            kind = type(item).__name__

            if kind == "Table":

                table = getattr(
                    item.metadata,
                    "text_as_html",
                    item.text
                )

                data["tables"].append(table)

    return data


def process_chunks(chunks):

    documents = []

    for chunk in chunks:

        data = separate_content(chunk)

        text = data["text"]

        tables = data["tables"]

        combined_content = f"""
{text}

TABLES:
{' '.join(tables)}
"""

        doc = Document(

            page_content=combined_content,

            metadata={
                "raw_text": text,
                "tables": str(tables),
                "source": "academic_pdf"
            }
        )

        documents.append(doc)

    return documents


def create_vectorstore(documents):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore


if __name__ == "__main__":

    file_path = "docs\\rag.pdf"

    print("Loading PDF...")

    elements = partition_document(file_path)

    print("Creating chunks...")

    chunks = chunk_document(elements)

    print("Processing chunks...")

    processed_documents = process_chunks(chunks)

    print("Generating embeddings and storing vectors...")

    vectorstore = create_vectorstore(processed_documents)

    print("Ingestion completed")


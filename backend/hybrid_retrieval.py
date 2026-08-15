import re
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

def tokenize_text(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())

def get_all_documents(vectorstore) -> list[Document]:
    try:
        data = vectorstore.get()
        documents = []
        if not data or "documents" not in data or not data["documents"]:
            return []
        
        for doc_text, metadata in zip(data["documents"], data["metadatas"]):
            documents.append(Document(page_content=doc_text, metadata=metadata or {}))
        return documents
    except Exception as e:
        print(f"Error fetching all documents from Chroma: {e}")
        return []

def hybrid_retrieve(vectorstore, query: str, k: int = 5) -> list[tuple[Document, float]]:
    query = query.strip()
    if not query:
        return []

    all_docs = get_all_documents(vectorstore)
    if not all_docs:
        print("No documents found in vectorstore to build BM25 index.")
        try:
            return vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"Fallback dense search failed: {e}")
            return []
   ## for dense we are still taking 2*k chunks  or 20 max of either of them
    dense_k = max(20, k * 2)
    try:
        dense_results = vectorstore.similarity_search_with_score(query, k=dense_k)
    except Exception as e:
        print(f"Dense search failed: {e}")
        dense_results = []

    dense_docs = [doc for doc, _ in dense_results]

    tokenized_corpus = [tokenize_text(doc.page_content) for doc in all_docs]
    tokenized_query = tokenize_text(query)
    
    sparse_docs = []
    if tokenized_query and tokenized_corpus:
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(tokenized_query)
        
        doc_scores = list(zip(all_docs, bm25_scores))
        relevant_doc_scores = [ds for ds in doc_scores if ds[1] > 0]
        relevant_doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        sparse_docs = [doc for doc, _ in relevant_doc_scores[:dense_k]]

    RRF_CONSTANT = 60
    rrf_scores = {}
    doc_map = {}

    def doc_key(doc: Document) -> str:
        return doc.page_content

    for rank, doc in enumerate(dense_docs, start=1):
        key = doc_key(doc)
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (RRF_CONSTANT + rank))

    for rank, doc in enumerate(sparse_docs, start=1):
        key = doc_key(doc)
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (RRF_CONSTANT + rank))

    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    fused_results = []
    for key in sorted_keys[:k]:
        fused_results.append((doc_map[key], rrf_scores[key]))

    return fused_results

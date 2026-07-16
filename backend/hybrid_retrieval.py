import re
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

def tokenize_text(text: str) -> list[str]:
    """
    Cleans and tokenizes text into lowercase alphanumeric words.
    """
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())

def get_all_documents(vectorstore) -> list[Document]:
    """
    Retrieves all stored documents from the Chroma vectorstore.
    Returns a list of LangChain Document objects.
    """
    try:
        # Chroma allows fetching all documents via get()
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
    """
    Performs Hybrid Retrieval by combining:
    1. Dense Search (Chroma vector search)
    2. Sparse Search (BM25 keyword search)
    
    Rankings from both are merged using Reciprocal Rank Fusion (RRF).
    Returns a list of tuples: (Document, rrf_score).
    """
    query = query.strip()
    if not query:
        return []

    # 1. Fetch all documents in the store to build BM25 corpus
    all_docs = get_all_documents(vectorstore)
    if not all_docs:
        print("No documents found in vectorstore to build BM25 index.")
        # Fall back to pure dense search
        try:
            return vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"Fallback dense search failed: {e}")
            return []

    # 2. Dense Search
    # Fetch more candidates than requested to allow effective RRF merging
    dense_k = max(20, k * 2)
    try:
        dense_results = vectorstore.similarity_search_with_score(query, k=dense_k)
    except Exception as e:
        print(f"Dense search failed: {e}")
        dense_results = []

    # Extract Document list from dense search results
    dense_docs = [doc for doc, _ in dense_results]

    # 3. Sparse Search (BM25)
    tokenized_corpus = [tokenize_text(doc.page_content) for doc in all_docs]
    tokenized_query = tokenize_text(query)
    
    # Check if we have tokens to query
    sparse_docs = []
    if tokenized_query and tokenized_corpus:
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Pair documents with their scores and sort descending
        doc_scores = list(zip(all_docs, bm25_scores))
        # Filter out documents with zero score to keep candidates relevant
        relevant_doc_scores = [ds for ds in doc_scores if ds[1] > 0]
        relevant_doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take up to dense_k candidates for RRF
        sparse_docs = [doc for doc, _ in relevant_doc_scores[:dense_k]]

    # 4. Reciprocal Rank Fusion (RRF)
    # RRF formula: RRF_score = sum(1 / (constant + rank))
    RRF_CONSTANT = 60
    rrf_scores = {}
    doc_map = {}

    def doc_key(doc: Document) -> str:
        # Use page_content as key for deduplication & mapping
        return doc.page_content

    # Process Dense Search ranks
    for rank, doc in enumerate(dense_docs, start=1):
        key = doc_key(doc)
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (RRF_CONSTANT + rank))

    # Process Sparse Search ranks
    for rank, doc in enumerate(sparse_docs, start=1):
        key = doc_key(doc)
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (RRF_CONSTANT + rank))

    # Sort candidates by their fused score in descending order
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Compile the final top-k results
    fused_results = []
    for key in sorted_keys[:k]:
        fused_results.append((doc_map[key], rrf_scores[key]))

    print(f"Hybrid retrieval finished: retrieved {len(dense_docs)} dense and {len(sparse_docs)} sparse docs. Combined into {len(fused_results)} top docs.")
    return fused_results

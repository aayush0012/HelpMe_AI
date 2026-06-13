
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


def load_retriever():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )

    return retriever


if __name__ == "__main__":

    retriever = load_retriever()

    llm = ChatOllama(
        model="llama3.2:latest"
    )

    query = input("Ask Question : ")

    results = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    prompt = f"""
```python id="jlwm106"
You are an academic assistant.

Answer the question strictly using the provided context.

Rules:
- Do not use outside knowledge
- If answer is not present, say:
  "Information not found in notes."
- Keep answers clear and accurate
- Preserve technical terminology
- If formulas are present, include them properly
```


CONTEXT:
{context}

QUESTION:
{query}
"""

    response = llm.invoke(prompt)

    print("\nAnswer:\n")

    print(response.content)


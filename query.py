import requests
import chromadb

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma"
CHAT_MODEL = "llama3.2:3b"
COLLECTION_NAME = "retail_knowledge"


def embed_texts(texts):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def ask_llama(prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
        },
    )
    response.raise_for_status()
    return response.json()["response"]


def main():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    count = collection.count()
    print(f"Collection has {count} documents")

    if count == 0:
        print("No documents found. Run ingest.py first.")
        return

    query = input("\nAsk a retail question: ").strip()

    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents"],
    )

    retrieved_docs = results["documents"][0]

    print("\nTOP MATCHES:\n")
    for idx, doc in enumerate(retrieved_docs, start=1):
        print(f"{idx}. {doc}")
        print()

    context = "\n".join(retrieved_docs)

    prompt = f"""
You are BuilderBot, a retail product assistant.

Answer the user's question using only the retrieved context below.

User question:
{query}

Retrieved context:
{context}

Rules:
- Use only the retrieved context.
- Start with a one-line direct answer.
- Then list matching products as bullet points.
- For each bullet, include:
  - product name
  - stock code if available
  - price if available
  - country if available
- Do not invent facts.
- Do not mention products that are not in the retrieved context.
"""

    answer = ask_llama(prompt)

    print("\nBuilderBot ANSWER:\n")
    print(answer)


if __name__ == "__main__":
    main()
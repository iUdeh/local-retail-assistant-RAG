import pandas as pd
import requests
import chromadb

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma"
COLLECTION_NAME = "retail_knowledge"


def embed_texts(texts):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def build_documents(limit=1500):
    df = pd.read_csv("data/online_retail.csv")
    df = df.dropna(subset=["Description", "Customer ID"])

    documents = []
    ids = []

    for i, row in df.head(limit).iterrows():
        description = str(row["Description"]).strip()

        doc = (
            f"Retail product record. "
            f"Product description: {description}. "
            f"This item was sold in {row['Country']}. "
            f"Invoice number: {row['Invoice']}. "
            f"Stock code: {row['StockCode']}. "
            f"Quantity sold: {row['Quantity']}. "
            f"Unit price: {row['Price']}. "
            f"Customer ID: {row['Customer ID']}. "
            f"Invoice date: {row['InvoiceDate']}. "
            f"Possible category hints come from the product description words."
        )

        documents.append(doc)
        ids.append(str(i))

    return ids, documents


def reset_collection(client):
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Old collection deleted.")
    except Exception:
        print("No old collection found.")


def main():
    client = chromadb.PersistentClient(path="./chroma_db")

    reset_collection(client)

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids, documents = build_documents(limit=1500)
    print(f"Created {len(documents)} documents")

    embeddings = embed_texts(documents)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
    )

    print("Documents stored in Chroma")
    print(f"Collection now has {collection.count()} documents")


if __name__ == "__main__":
    main()
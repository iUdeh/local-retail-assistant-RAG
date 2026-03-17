import requests
import chromadb
import streamlit as st

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma"
CHAT_MODEL = "llama3.2:3b"
COLLECTION_NAME = "retail_knowledge"


def embed_texts(texts):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
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
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name=COLLECTION_NAME)


st.set_page_config(page_title="Retail RAG Assistant", page_icon="🛍️", layout="wide")

collection = get_collection()
count = collection.count()

with st.sidebar:
    st.header("Project Info")
    st.write(f"**Chat model:** `{CHAT_MODEL}`")
    st.write(f"**Embedding model:** `{EMBED_MODEL}`")
    st.write(f"**Collection:** `{COLLECTION_NAME}`")
    st.write(f"**Indexed docs:** `{count}`")

    st.divider()

    st.header("Sample Questions")
    st.markdown(
        """
- Which products are related to Christmas?
- Which items are suitable for kitchen use?
- Which products seem decorative?
- What items are related to baking or cooking?
- What products can I use for decorations?
        """
    )

    st.divider()

    st.header("How to Use")
    st.markdown(
        """
1. Type a retail question  
2. Choose how many matches to retrieve  
3. Click **Ask BuilderBot**
        """
    )

st.title("🛍️ Your Local Retail Assistant")
st.caption("Ask questions about your retail dataset using Ollama + Chroma")

st.write(f"Indexed documents: **{count}**")

if count == 0:
    st.warning("No documents found. Run `python ingest.py` first.")
    st.stop()

query = st.text_input(
    "Ask a retail question",
    placeholder="e.g. Which products are related to Christmas?"
)

top_k = st.slider("Number of matches", min_value=3, max_value=10, value=5)

if st.button("Ask BuilderBot"):
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Retrieving relevant products..."):
        query_embedding = embed_texts([query])[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents"],
        )

        retrieved_docs = results["documents"][0]
        context = "\n".join(retrieved_docs)

    st.subheader("Top Matches")
    for idx, doc in enumerate(retrieved_docs, start=1):
        with st.expander(f"Match {idx}", expanded=(idx == 1)):
            st.write(doc)

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

    with st.spinner("Generating answer..."):
        answer = ask_llama(prompt)

    st.subheader("BuilderBot Answer")
    st.write(answer)
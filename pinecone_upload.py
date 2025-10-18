# pinecone_upload.py
import json
import time
from tqdm import tqdm
## from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import config
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32
## OPENAI_EMBEDDING_MODEL = "text-embedding-3-small" ##added the configureable values in the config file rather than hardcoding it in the embedding file
INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = config.PINECONE_VECTOR_DIM  # 1536 for text-embedding-3-small

# -----------------------------
# Initialize clients
# -----------------------------
# client = OpenAI(api_key=config.OPENAI_API_KEY) ## no more using openai client, using sentence transformers locally
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# -----------------------------
# Initialize local embedding model
# -----------------------------
print("using a local embedding model rather than openai api calls")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # 384 dimensional embeddings

# -----------------------------
# Create managed index if it doesn't exist
# -----------------------------
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
    name=INDEX_NAME,
    dimension=VECTOR_DIM,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws", ##had to change it o aws from gcp as it was a free server
        region="us-east-1"
    )
)
else:
    print(f"Index {INDEX_NAME} already exists.")

# Connect to the index
index = pc.Index(INDEX_NAME)

# -----------------------------
# Helper functions
# -----------------------------
##the previous script would have crahsed if one openai api request fails as it had no erro handling
##def get_embeddings(texts, model=OPENAI_EMBEDDING_MODEL): ##un hardcoded it 
##    """Generate embeddings using OpenAI v1.0+ API."""
def get_embeddings(texts):
    return model.encode(texts, show_progress_bar=True).tolist()
    #try:    ##implemented execption handling
    #    resp = client.embeddings.create(model=model, input=texts)
    #    return [data.embedding for data in resp.data]
    #except Exception as e:
    #    print(f"An error occurred while getting embeddings: {e}")
    #    # Decide how to handle the error, return empty embeddings or raise an error
    #    # since the number of empty lists must match the number of input texts if its sparse
    #    return [[] for _ in texts] 
    
def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        items.append((node["id"], semantic_text, meta))

    print(f"Preparing to upsert {len(items)} items to Pinecone...")

    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = get_embeddings(texts) ##removed the hardcoded model name from the function call

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        index.upsert(vectors)
        time.sleep(0.2)

    print("All items uploaded successfully.")

# -----------------------------
if __name__ == "__main__":
    main()

'''
import json
import time
from tqdm import tqdm
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import config

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32
VECTOR_DIM = config.PINECONE_VECTOR_DIM  # 1536 for text-embedding-3-small
INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize clients
# -----------------------------
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# -----------------------------
# Create managed index if it doesn't exist
# -----------------------------
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Index {INDEX_NAME} already exists.")

# Connect to the index
index = pc.Index(INDEX_NAME)

# -----------------------------
# Helper functions
# -----------------------------
def get_embeddings(texts, model="text-embedding-3-small"):
    """Generate mock embeddings."""
    np.random.seed(42)
    return [np.random.rand(VECTOR_DIM).tolist() for _ in texts]

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        items.append((node["id"], semantic_text, meta))

    print(f"Preparing to upsert {len(items)} items to Pinecone...")

    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = get_embeddings(texts)

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        index.upsert(vectors)
        time.sleep(0.2)

    print("All items uploaded successfully.")

# -----------------------------
if __name__ == "__main__":
    main()'''
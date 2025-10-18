#hybrid_chat.py
"""
An interactive command-line interface for the Hybrid AI Travel Assistant.
This script integrates a local sentence transformer for embeddings, a local Llama 3
model for reasoning, Pinecone for semantic search, and Neo4j for graph context
to provide intelligent answers for travel-related queries about Vietnam.

An assignment submission for the AI Engineer role at Blue Enigma.

Author: Mudassir Alam N
Date: 2025-10-18
"""
import json
import asyncio
from typing import List
import torch
from pinecone import Pinecone
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import config

# -----------------------------
# Use MPS for GPU acceleration if available
# -----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device for acceleration.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

# -----------------------------
# Config
# -----------------------------
TOP_K = config.PINECONE_TOP_K
INDEX_NAME = config.PINECONE_INDEX_NAME
EMBED_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
RELEVANCE_THRESHOLD = 0.4 # Threshold for filtering irrelevant results

# -----------------------------
# Initialize clients
# -----------------------------
print("Loading local chat model (Llama 3 8B)...")
llm = Llama.from_pretrained(
    repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
    filename="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    verbose=False
)
pc = Pinecone(api_key=config.PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
driver = GraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# -----------------------------
# Helper functions
# -----------------------------
_embedding_cache = {}
def embed_text(text: str) -> List[float]:
    """Generate or retrieve a cached embedding for a text string."""
    if text in _embedding_cache:
        return _embedding_cache[text]
    embedding = EMBED_MODEL.encode(text).tolist()
    _embedding_cache[text] = embedding
    return embedding

async def pinecone_query(query_text: str, top_k=TOP_K):
    """Asynchronously query Pinecone by running the sync method in a thread."""
    vec = embed_text(query_text)
    return await asyncio.to_thread(
        index.query,
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )

def _sync_fetch_graph_context(node_ids: List[str]):
    """Synchronous helper function to fetch data from Neo4j."""
    facts = []
    with driver.session() as session:
        for nid in node_ids:
            q = ("MATCH (n {id:$nid})-[r]-(m) "
                 "RETURN type(r) AS rel, m.id AS id, m.name AS name LIMIT 5")
            recs = session.run(q, nid=nid)
            for r in recs:
                facts.append({"source": nid, "rel": r["rel"], "target_id": r["id"], "target_name": r["name"]})
    return facts

async def fetch_graph_context(node_ids: List[str]):
    """Asynchronously fetch Neo4j context by running the sync helper in a thread."""
    return await asyncio.to_thread(_sync_fetch_graph_context, node_ids)

def summarize_context(context: str) -> str:
    """Uses the LLM to summarize a block of text."""
    messages = [
        {"role": "system", "content": "You are a text summarization assistant. Summarize the following context into a concise paragraph, focusing on key entities and their relationships."},
        {"role": "user", "content": context}
    ]
    return get_llm_response(messages)

def build_prompt(user_query: str, summarized_context: str) -> list:
    """Builds the final prompt for the LLM using the summarized context."""
    system_prompt = "You are a helpful travel assistant. Answer the user's question based ONLY on the summarized context provided."
    user_prompt = f"Summarized Context:\n{summarized_context}\n\nQuestion: {user_query}"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

def get_llm_response(messages: list) -> str:
    """Generates a response using the local Llama 3 model."""
    try:
        response = llm.create_chat_completion(messages=messages)
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return "Sorry, I encountered an error while generating the response."

# -----------------------------
# Interactive chat
# -----------------------------
async def interactive_chat():
    """Main async loop for the interactive chat assistant."""
    print("\nHybrid travel assistant. Type 'exit' to quit.")
    while True:
        query = await asyncio.to_thread(input, "\nEnter your travel question: ")
        query = query.strip()
        if not query or query.lower() in ("exit", "quit"):
            print("Exiting...")
            break

        pinecone_res = await pinecone_query(query, top_k=TOP_K)
        matches = pinecone_res["matches"]

        if matches:
            print(f"DEBUG: Top match score is {matches[0].get('score', 0.0)}")

        if not matches or matches[0].get('score', 0.0) < RELEVANCE_THRESHOLD:
            print("\n=== Assistant Answer ===\nI'm sorry, but I don't have enough information about that topic to provide a helpful answer.\n======================\n")
            continue

        match_ids = [m["id"] for m in matches]
        graph_facts = await fetch_graph_context(match_ids)

        raw_context = ""
        for match in matches[:3]:
            meta = match.get('metadata', {})
            raw_context += f"- A {meta.get('type', 'place')} called '{meta.get('name', 'N/A')}' is in {meta.get('city', 'N/A')}.\n"
        for fact in graph_facts[:5]:
            raw_context += f"- The item '{fact['source']}' is related to '{fact['target_name']}'.\n"

        print("\n--- Summarizing Retrieved Context ---")
        summarized_context = summarize_context(raw_context)
        print(f"DEBUG: Summarized Context: {summarized_context}")

        final_prompt_messages = build_prompt(query, summarized_context)
        print("\n--- Generating Final Answer ---")
        final_answer = get_llm_response(final_prompt_messages)
        print("\n=== Assistant Answer ===\n" + final_answer + "\n======================\n")

if __name__ == "__main__":
    try:
        asyncio.run(interactive_chat())
    except KeyboardInterrupt:
        print("\nExiting...")
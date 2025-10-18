Hybrid AI Travel Assistant Challenge Submission
Submitted by: Mudassir Alam N Date: 2025-10-18

1. Project Overview
This repository contains a solution to the Blue Enigma AI Engineer Technical Challenge. The project is a sophisticated, fully self-hosted hybrid AI assistant that answers travel queries about Vietnam.
It implements an advanced Retrieval-Augmented Generation (RAG) architecture, retrieving context from a vector database (Pinecone) and a knowledge graph (Neo4j) simultaneously. The reasoning and embedding layers run entirely on local hardware, leveraging a state-of-the-art quantized Llama 3 model for intelligent responses.
For a detailed narrative of the development process, architectural decisions, and challenges overcome, please see the improvements.md report.

2. Key Features & Architecture
Fully Self-Hosted: No reliance on paid external APIs. Uses a local SentenceTransformer for embeddings and a local Llama 3 8B model for reasoning.
Hybrid Retrieval: Fetches semantic context from Pinecone and relational graph facts from Neo4j.
Advanced RAG Pipeline: Implements a two-step "Summarize-then-Answer" (Chain-of-Thought) process to improve the quality of generated answers.
High Performance:
Asynchronous I/O: Pinecone and Neo4j queries run in parallel using asyncio to reduce latency.
Hardware Acceleration: Utilizes Apple Silicon's Metal GPU (MPS) via llama-cpp-python for fast LLM inference.
Embedding Caching: An in-memory cache avoids redundant embedding calculations.
Robustness: A "Relevance Gate" checks the similarity score of retrieved results to prevent the system from answering out-of-scope questions and reduce hallucinations.

3. Setup and Installation
Prerequisites
Python 3.10+
An active internet connection (for the initial model download)
Accounts for Pinecone and Neo4j (AuraDB free tier is sufficient)
Step 1: Clone the Repository
git clone [https://github.com/mudxssir/HybridAi.git](https://github.com/mudxssir/HybridAi.git)
cd blue-enigma-hybrid-ai-challenge

Step 2: Create and Activate a Virtual Environment
python3 -m venv venv
source venv/bin/activate

Step 3: Install Dependencies
This single command will install all necessary libraries. The CMAKE_ARGS are critical for enabling GPU acceleration on Apple Silicon (M1/M2/M3) machines.
CMAKE_ARGS="-DLLAMA_METAL=on" pip install --no-cache-dir -r requirements.txt

(Note: A requirements.txt file should be created containing neo4j, pinecone-client, sentence-transformers, torch, and llama-cpp-python)
Step 4: Configure API Keys
Copy the sample configuration file and fill in your credentials.
cp config.py.sample config.py

Now, edit config.py and add your keys for Pinecone and Neo4j.
4. How to Run the System
Execute the following scripts from your terminal in order.
Step 1: Load Data into Neo4j
This script populates your Neo4j database with the provided dataset.
python3 load_to_neo4j.py

Step 2: Upload Embeddings to Pinecone
This script generates embeddings locally and uploads them to your Pinecone index.
python3 pinecone_upload.py

Step 3: Start the Interactive Chat
This will start the AI assistant. The first time you run this, it will download the ~5GB Llama 3 model, which may take some time.
python3 hybrid_chat.py

You can then ask questions like:
create a romantic 4 day itinerary for Vietnam
What is near the Da Nang Hotel?
What are the best ski resorts in Switzerland? (to test the relevance gate)



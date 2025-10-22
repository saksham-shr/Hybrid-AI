HybridAI Documentation
A Self‑Hosted Hybrid RAG Chatbot using Llama 3, Neo4j, and Pinecone

1. Introduction
HybridAI is an end‑to‑end retrieval‑augmented generation (RAG) assistant built for travel domain reasoning using a hybrid data pipeline — combining vector embeddings (Pinecone), knowledge graphs (Neo4j), and a quantized Llama 3 8B Instruct model.

It operates fully offline, optimized for consumer hardware and designed for real‑world efficiency, scalability, and transparency.

2. Key Features
Self‑hosted hybrid architecture (no paid APIs)

RAG pipeline integrating semantic and relational retrieval

Meta Llama 3 8B Instruct (Q4_K_M) quantized model for efficient inference

Asynchronous I/O and embedding cache for faster responses

Relevance Gate (0.4) to reduce hallucinations

Two‑stage reasoning pipeline – context summarization → final answer

Local Vietnamese travel knowledge graph and dataset

3. System Requirements
Component	Minimum Requirement
Python	3.10 or newer
RAM	 8 GB+ (16 GB recommended)
GPU	Apple M1/M2 (MPS) or NVIDIA CUDA support
Databases	Neo4j 5.x, Pinecone Cloud Index
OS	macOS, Linux, Windows 10+
4. Installation
4.1 Clone Repository
bash
git clone https://github.com/mudxssir/HybridAI.git
cd HybridAI
4.2 Create and Activate Virtual Environment
bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
4.3 Install Dependencies
bash
pip install -r requirements.txt
5. Configuration
Copy the sample config and update with your credentials.

bash
cp config.py.sample config.py
Edit config.py:

python
NEO4J_URI = "neo4j+s://your-db-uri.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

PINECONE_API_KEY = "your_pinecone_api_key"
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_VECTOR_DIM = 384
PINECONE_TOP_K = 5
6. Data Pipeline Setup
6.1 Load Travel Data to Neo4j
bash
python load_to_neo4j.py
6.2 Upload Vector Embeddings to Pinecone
bash
python pinecone_upload.py
Embeddings (384‑dim MiniLM‑L6‑v2) are generated locally and uploaded to the configured index.

7. Visualization
Generate and view the graph relations:

bash
python visualize_graph.py
Result is saved as neo4j_viz.html.

8. Running the Chatbot
Launch the interactive CLI chat:

bash
python hybrid_chat.py
Sample:

text
Hybrid travel assistant. Type 'exit' to quit.
Enter your travel question: What are the cultural attractions in Hanoi?
9. Architecture Overview
Flowchart Overview:

Query → Embeddings via SentenceTransformers

Search context from Pinecone (top‑k semantic matches)

Enrich context from Neo4j (related entity graph)

Summarize retrieved data using Llama 3

Generate final grounded response

This hybrid fusion of semantic and symbolic reasoning improves accuracy and reduces hallucination.

10. Deployment on a Website
To deploy HybridAI on a site (such as offbeatsikkim.com):

10.1 Build a Flask Backend API
python
from flask import Flask, request, jsonify
import asyncio
from hybrid_chat import get_chat_response

app = Flask(__name__)

@app.route('/api/chat', methods=['POST'])
def chat():
    msg = request.json.get('message', '')
    answer = asyncio.run(get_chat_response(msg))
    return jsonify({'response': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
10.2 Embed Frontend Widget
xml
<div id="chatbox"></div>
<script>
async function sendMessage(msg){
  const res = await fetch("https://your-domain.com/api/chat",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({message:msg})
  });
  const data = await res.json();
  console.log(data.response);
}
</script>
10.3 Deploy
Use a VPS, AWS EC2, or Render Service to host backend

Point API to your domain and enable HTTPS

Add JS snippet site‑wide for live chatbot access

11. Directory Structure
text
HybridAI/
├── hybrid_chat.py
├── pinecone_upload.py
├── load_to_neo4j.py
├── visualize_graph.py
├── vietnam_travel_dataset.json
├── config.py.sample
├── requirements.txt
├── lib/
└── README.md
12. Tech Stack
Layer	Technology
LLM	Meta Llama 3 8B Instruct (Q4_K_M)
Embeddings	Sentence‑Transformers (all‑MiniLM‑L6‑v2)
Vector Search	Pinecone v2
Knowledge Graph	Neo4j 5.x
Visualization	PyVis + NetworkX
Backend	Python (Flask)
13. Future Enhancements
Extend dataset to cover broader geographies (e.g., Sikkim Tourism)

Streamlit / React web UI

Multi‑turn memory for contextual conversation

Dockerized microservice for cloud deployment

Plugin‑style API for custom travel data

14. License
MIT License
Permission is granted to use, copy, and modify this project for research or commercial purposes with attribution.

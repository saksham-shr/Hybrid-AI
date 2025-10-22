# **HybridAI Documentation**
### *A Self-Hosted Hybrid RAG Chatbot using Llama 3, Neo4j, and Pinecone*

---

## **1. Introduction**
**HybridAI** is an end-to-end **Retrieval-Augmented Generation (RAG)** assistant designed for **travel domain reasoning**, combining the strengths of **vector embeddings** (Pinecone), **knowledge graphs** (Neo4j), and a **quantized Llama 3 8B Instruct model**.

It is fully **self-hosted**, optimized for **consumer hardware**, and focuses on **efficiency, scalability, and transparency** — enabling offline operation without dependence on paid APIs.

---

## **2. Key Features**
- 🧠 **Hybrid Architecture:** Combines semantic (vector) and relational (graph) retrieval.  
- 💻 **Self-Hosted:** No paid API dependencies.  
- ⚡ **Efficient Inference:** Uses quantized **Meta Llama 3 8B Instruct (Q4_K_M)**.  
- 🔄 **Optimized Pipeline:** Asynchronous I/O and embedding cache for faster responses.  
- 🧩 **Relevance Gate (0.4):** Minimizes hallucinations and improves precision.  
- 🧠 **Two-Stage Reasoning:** Context summarization → grounded answer generation.  
- 🌏 **Localized Dataset:** Includes Vietnamese travel knowledge graph and dataset.  

---

## **3. System Requirements**

| **Component** | **Minimum Requirement** |
|----------------|--------------------------|
| **Python** | 3.10 or newer |
| **RAM** | 8 GB+ (16 GB recommended) |
| **GPU** | Apple M1/M2 (MPS) or NVIDIA CUDA support |
| **Databases** | Neo4j 5.x, Pinecone Cloud Index |
| **OS** | macOS, Linux, Windows 10+ |

---

## **4. Installation**

### **4.1 Clone Repository**
```bash
git clone https://github.com/saksham-shr/Hybrid-AI.git
cd Hybrid-AI
```

### **4.2 Create and Activate Virtual Environment**
```bash
python3 -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### **4.3 Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **5. Configuration**

Copy the sample configuration file and update your credentials:
```bash
cp config.py.sample config.py
```

Edit `config.py`:
```python
NEO4J_URI = "neo4j+s://your-db-uri.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

PINECONE_API_KEY = "your_pinecone_api_key"
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_VECTOR_DIM = 384
PINECONE_TOP_K = 5
```

---

## **6. Data Pipeline Setup**

### **6.1 Load Travel Data into Neo4j**
```bash
python load_to_neo4j.py
```

### **6.2 Upload Vector Embeddings to Pinecone**
```bash
python pinecone_upload.py
```
- Embeddings (384-dim) are generated locally using `MiniLM-L6-v2`.  
- Data is then uploaded to the configured Pinecone index.  

---

## **7. Visualization**

Generate and visualize the travel knowledge graph:
```bash
python visualize_graph.py
```
Output: `neo4j_viz.html` — an interactive graph visualization.

---

## **8. Running the Chatbot**

Launch the interactive CLI chatbot:
```bash
python hybrid_chat.py
```

**Sample:**
```
Hybrid travel assistant. Type 'exit' to quit.
Enter your travel question: What are the cultural attractions in Hanoi?
```

---

## **9. Architecture Overview**

**Pipeline Flow:**
1. Query → Embeddings via SentenceTransformers  
2. Retrieve top-k semantic matches from Pinecone  
3. Enrich results with related entities from Neo4j  
4. Summarize retrieved data using Llama 3  
5. Generate a grounded final response  

This **hybrid fusion of semantic and symbolic reasoning** enhances factual accuracy and reduces hallucinations.

---

## **10. Deployment on a Website**

### **10.1 Build Flask Backend API**
```python
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
```

### **10.2 Embed Frontend Widget**
```html
<div id="chatbox"></div>
<script>
async function sendMessage(msg){
  const res = await fetch("https://your-domain.com/api/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ message: msg })
  });
  const data = await res.json();
  console.log(data.response);
}
</script>
```

### **10.3 Deploy**
- Use a VPS, AWS EC2, or Render Service for hosting.  
- Point the API endpoint to your domain and enable HTTPS.  
- Embed the JS snippet across your site for chatbot access.  

---

## **11. Directory Structure**
```
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
```

---

## **12. Tech Stack**

| **Layer** | **Technology** |
|------------|----------------|
| **LLM** | Meta Llama 3 8B Instruct (Q4_K_M) |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) |
| **Vector Search** | Pinecone v2 |
| **Knowledge Graph** | Neo4j 5.x |
| **Visualization** | PyVis + NetworkX |
| **Backend** | Python (Flask) |

---

## **13. Future Enhancements**
- 🌍 Extend dataset to include broader geographies (e.g., **Sikkim Tourism**).  
- 💬 Build **Streamlit** or **React** web UI.  
- 🧠 Add **multi-turn conversation memory**.  
- 🐳 Dockerize as a microservice for cloud deployment.  
- 🔌 Develop **plugin-style API** for custom travel data sources.  

---

## **14. License**
**MIT License**  

This software is for personal, educational, or research purposes only.  
Any commercial or redistribution use requires prior written consent from the author.


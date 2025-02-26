# 🧠 Local Single AI Agent with Integrated RAG  

This repository hosts a **local AI-powered Retrieval-Augmented Generation (RAG) system** that runs entirely on your machine. It is a **single-agent AI** that retrieves information from indexed documents and enhances responses using a locally running **LLM (Llama3.2 or DeepSeek) with Nomic-Embed-Text** for embedding generation.  

---

## 📂 Folder Structure  

![image](https://github.com/user-attachments/assets/84e75374-86a9-425f-820d-b2ae18f7cec9)


🛠 Updated Features
- graph_rag/ → Handles Graph-based RAG operations
- graph_rag/index.py → Contains logic for document graph indexing
- graph_rag/index.json → Stores chunk embeddings in JSON format
- app.py → Can generate an interactive visualization of RAG as an HTML graph
- README.md → Will guide users on how to set up & run the system


---

## ⚡ Features  

✅ **Runs Locally** – No external API calls, ensuring full privacy  
✅ **Multi-Model Support** – Choose between `Llama3.2:latest` and `DeepSeek-r1:14b` for generating responses  
✅ **Nomic-Embed-Text** – Generates high-quality embeddings for better retrieval  
✅ **Graph-Based RAG Visualization** – Displays indexed documents and chunks as an interactive **Graph Network**  
✅ **Time Tracking** – Tracks retrieval, embedding, response generation, and total processing time  

---

## 🛠 System Requirements  

To run the **Local AI Agent with RAG**, your system should have:  

- **Minimum VRAM:** **8GB** (Dedicated GPU recommended)  
- **Supported GPUs:** NVIDIA RTX 2070 or higher (CUDA enabled)  
- **CPU:** At least **6-core** processor for smooth inference  
- **RAM:** **16GB+** recommended for processing large documents  
- **Storage:** Sufficient space for storing indexed documents and embeddings  

---

## 🎨 Graph Visualization  

The app generates a **graph-based visualization** of indexed documents and chunks.  

🔹 **Documents** are represented as **blue nodes**  
🔹 **Chunks** are **green nodes**, connected to their respective documents  
🔹 **Similarity edges** are drawn between related chunks  

### How to View the Graph?  

1. Run `app.py` and use the sidebar option **Visualize Graph RAG**  
2. The HTML file `graph.html` will be generated  
3. Open `graph.html` separately in a browser  

> **Note:** Uncomment and modify `app.py` as needed to enhance the visualization.  

---

## 🔧 Enhancements & Customization  

- Adjust document **chunking strategy** in `chunking.py`  
- Modify **retrieval strategy** in `index.py`  
- Extend **graph visualization** in `app.py`  

This project is fully customizable—build your own **private local AI agent** using RAG! 🚀  

---

## 📢 Contribution

Pull requests and feature suggestions are welcome! 🙌  

---

## 📜 License  

MIT License © 2025 Your Name  

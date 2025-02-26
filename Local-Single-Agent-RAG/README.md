# 🧠 Local Single AI Agent with Integrated RAG  

This repository hosts a **local AI-powered Retrieval-Augmented Generation (RAG) system** that runs entirely on your machine. It is a **single-agent AI** that retrieves information from indexed documents and enhances responses using a locally running **LLM (Llama3.2 or DeepSeek) with Nomic-Embed-Text** for embedding generation.  

![App Snapshot](https://github.com/itsual/AI-Agents/blob/main/Local-Single-Agent-RAG/LocalRAG-1.gif)


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

## Two cents

Here’s a list of practical tips based on my journey building a local single-agent RAG system:

1. Plan Your Workflow:  
   Outline every step—from PDF extraction and text chunking to embedding, indexing, retrieval, and LLM response generation. A clear workflow helps you pinpoint where issues arise.

2. Test Modules Independently:  
   Before integrating, verify that each module (PDF extraction, embedding, LLM query) works in isolation. For example, test your document converter in a Jupyter Notebook to confirm it outputs the expected Markdown.

3. Mind Your Data Shapes:  
   Ensure that embeddings have consistent shapes. A shape mismatch (e.g., (1,768) vs. (1,1)) can break cosine similarity calculations. Always reshape and validate vectors before performing arithmetic operations.

4. Robust Indexing:  
   Store both the raw text and embeddings for each chunk in your index (e.g., an `index.json` file). Make sure your indexing script correctly appends new documents and handles duplicates or updates gracefully.

5. Monitor Performance:  
   Measure time for extraction, embedding, and response generation. Adding timing logs helps identify bottlenecks so you can optimize or adjust your hardware expectations.

6. Graph Visualization:  
   When visualizing your RAG graph, check that node attributes (like “group” or “label”) are consistently set. If errors arise, try simplifying the node data or handling missing attributes.

7. Error Handling & Logging:  
   Implement detailed error logging. Whether it’s a failure in document extraction, embedding generation, or graph rendering, having meaningful error messages is key to rapid troubleshooting.

8. Resource Management:  
   If you’re running inference locally, ensure your GPU resources are effectively utilized. Monitor VRAM usage and consider offloading some layers to CPU if needed.

9. Prompt Engineering:  
   Construct clear and concise prompts for the LLM. Experiment with different prompt templates to get more relevant responses.

10. Iterative Development:  
    Build incrementally. Start with a simple prototype (like a basic chatbot) before adding advanced features such as graph visualization or multi-step retrieval.


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

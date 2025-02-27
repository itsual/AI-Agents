import streamlit as st
import requests
import os
import json
import time
import numpy as np
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from sklearn.metrics.pairwise import cosine_similarity

from utils.pdf_utils import pdf_to_markdown
from utils.embedding import get_embedding_for_text
from utils.local_embeddings_llm import generate_response
from utils.chunking import chunk_text

# Set page config
st.set_page_config(page_title="Multi Agent AI Bot", layout="wide")

# Global configuration
LLM_PROXY_URL = "http://127.0.0.1:11434"
INPUT_FOLDER = "input"
INDEX_FILE = "index.json"

if not os.path.exists(INPUT_FOLDER):
    os.makedirs(INPUT_FOLDER)

# -------------------------------
# Helper Functions
# -------------------------------

def cosine_similarity_custom(vec1, vec2):
    """Compute cosine similarity between two 1D numpy arrays."""
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    if vec1.shape != vec2.shape:
        return 0
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

def retrieve_context(query, index_file=INDEX_FILE):
    """Retrieve the most relevant chunk based on cosine similarity."""
    query_embedding = get_embedding_for_text(query)
    if not query_embedding:
        return "No relevant document context found."
    query_embedding = np.array(query_embedding)
    if not os.path.exists(index_file):
        return "No relevant document context found."
    with open(index_file, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    best_similarity = -1
    best_text = "No relevant document context found."
    for doc in index_data:
        for chunk in doc.get("chunks", []):
            emb = np.array(chunk.get("embedding", []))
            if emb.size > 0 and emb.shape == query_embedding.shape:
                sim = cosine_similarity_custom(query_embedding, emb)
                if sim > best_similarity:
                    best_similarity = sim
                    best_text = chunk.get("chunk_text", "")
    return best_text

def query_knowledge_graph(question, temperature, max_tokens, model):
    """Retrieve context, construct prompt, and measure processing times."""
    start_total = time.time()
    start_retrieval = time.time()
    context = retrieve_context(question)
    retrieval_time = time.time() - start_retrieval

    prompt = (
        f"Using the following context:\n{context}\n\n"
        f"Answer the question:\n{question}\n\n"
        "Provide a clear answer."
    )

    start_embedding = time.time()
    _ = get_embedding_for_text(question)  # For timing purposes only
    embedding_time = time.time() - start_embedding

    start_response = time.time()
    response = generate_response(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    response_time = time.time() - start_response

    total_time = time.time() - start_total

    st.write("⏳ **Timing Details**")
    st.write(f"- ⏱ Retrieval Time: `{retrieval_time:.2f} sec`")
    st.write(f"- ⏳ Embedding Time: `{embedding_time:.2f} sec`")
    st.write(f"- 🤖 LLM Response Time: `{response_time:.2f} sec`")
    st.write(f"- 🏁 Total Processing Time: `{total_time:.2f} sec`")
    return response


def build_graph(index_file="index.json", threshold=0.5):
    """
    Builds a NetworkX graph with:
      - A node for each document
      - A node for each chunk within a document
      - An edge from a document to each of its chunks
      - (Optional) edges between chunks across documents if they exceed 'threshold' similarity
    """
    if not os.path.exists(index_file):
        return nx.Graph()

    with open(index_file, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    G = nx.Graph()

    # 1) Add document nodes and their chunk nodes
    for doc in index_data:
        doc_id = doc["filename"]
        # Create a node for the document
        G.add_node(
            doc_id,
            label=doc_id,   # or some short label
            group="document",
            size=30
        )

        # Add each chunk as a separate node
        for i, chunk in enumerate(doc.get("chunks", [])):
            chunk_id = f"{doc_id}__chunk_{i}"
            chunk_label = f"Chunk {i}: {chunk['chunk_text'][:50]}..."
            # Create chunk node
            G.add_node(
                chunk_id,
                label=chunk_label,
                group="chunk",
                size=10
            )
            # Add edge from doc node to chunk node
            G.add_edge(doc_id, chunk_id, weight=1.0)

    # 2) (Optional) Compare chunk-to-chunk across different docs
    #    to add edges if similarity >= threshold
    #    If you only want doc-level edges, you can skip this step
    #    But if you want chunk-level cross-links, do the following:
    all_chunks = []
    for doc in index_data:
        doc_id = doc["filename"]
        for i, chunk in enumerate(doc.get("chunks", [])):
            chunk_id = f"{doc_id}__chunk_{i}"
            emb = chunk.get("embedding", [])
            all_chunks.append((chunk_id, emb))

    for i in range(len(all_chunks)):
        for j in range(i + 1, len(all_chunks)):
            chunk_id1, emb1 = all_chunks[i]
            chunk_id2, emb2 = all_chunks[j]

            if not emb1 or not emb2:
                continue  # Skip empty embeddings

            # Convert to numpy arrays
            emb1_arr = np.array(emb1)
            emb2_arr = np.array(emb2)
            if emb1_arr.size == 0 or emb2_arr.size == 0:
                continue

            sim = cosine_similarity(emb1_arr, emb2_arr)
            if sim >= threshold:
                # Add an edge for chunk-chunk similarity
                G.add_edge(chunk_id1, chunk_id2, weight=sim)

    return G

def visualize_graph(G):
    """Visualizes the graph using PyVis and returns the HTML string."""
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    return net.generate_html()

# -------------------------------
# Sidebar Controls
# -------------------------------

# Button to clear the index file
if st.sidebar.button("Clear Index"):
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
        st.sidebar.success("Index file cleared.")
        st.rerun()
    else:
        st.sidebar.info("Index file does not exist.")

st.sidebar.header("Settings")
temperature = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.7, 0.05)
max_tokens = st.sidebar.number_input("Max Tokens", 50, 1024, 256, 50)
llm_model = st.sidebar.selectbox("LLM Model", ["llama3.2:latest", "deepseek-r1:14b"])
if st.sidebar.button("Clear Conversation"):
    if "conversation" in st.session_state:
        del st.session_state["conversation"]
    st.rerun()

if st.sidebar.button("Visualize Graph RAG"):
    G = build_graph()
    if G.number_of_nodes() == 0:
        st.sidebar.info("No documents indexed.")
    else:
        graph_html = visualize_graph(G)
        components.html(graph_html, height=600, scrolling=True)

# -------------------------------
# PDF Upload & Indexing
# -------------------------------

st.header("Upload PDF Document and Update Index")
uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_pdf is not None:
    file_name = uploaded_pdf.name
    file_path = os.path.join(INPUT_FOLDER, file_name)
    
    # Check if the document is already processed
    already_processed = False
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        for doc in idx_data:
            if doc.get("filename", "") == file_name:
                already_processed = True
                break
    else:
        idx_data = []

    if already_processed:
        st.info(f"Document '{file_name}' is already processed. Skipping extraction and embedding.")
    else:
        start_time = time.time()
        with open(file_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        st.success(f"Uploaded {file_name} successfully!")
        
        md_text = pdf_to_markdown(file_path)
        if md_text:
            extraction_time = time.time() - start_time
            st.markdown(f"<p style='color:blue;'>Extraction completed in {extraction_time:.2f} seconds.</p>", unsafe_allow_html=True)
            show_extracted = st.checkbox("Show Extracted Text")
            if show_extracted:
                st.markdown(md_text)
            # Process document: chunk and embed
            chunks = chunk_text(md_text, method="fixed_window", max_words=1000, overlap=200)
            doc_record = {"filename": file_name, "text": md_text, "chunks": []}
            embed_start = time.time()
            for c in chunks:
                emb = get_embedding_for_text(c)
                doc_record["chunks"].append({"chunk_text": c, "embedding": emb})
            embed_time = time.time() - embed_start
            st.markdown(f"<p style='color:blue;'>Embedding generation completed in {embed_time:.2f} seconds.</p>", unsafe_allow_html=True)
            idx_data.append(doc_record)
            with open(INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump(idx_data, f, ensure_ascii=False, indent=2)
            st.success("Index updated with the document.")
        else:
            st.error("Failed to extract content from the PDF.")

# -------------------------------
# Chat Interface
# -------------------------------

st.title("Multi Agent AI Bot")
st.markdown("This application retrieves context from indexed documents and uses a local LLM to generate responses.")
if "conversation" not in st.session_state:
    st.session_state.conversation = []

with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for entry in st.session_state.conversation:
        if entry["sender"] == "user":
            st.markdown(f'<div class="user-msg"><strong>User:</strong> {entry["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg"><strong>Bot:</strong> {entry["text"]}</div>', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Enter your question:")
        submit_button = st.form_submit_button("Send")
    if submit_button and user_input:
        st.session_state.conversation.append({"sender": "user", "text": user_input})
        with st.spinner("Generating answer..."):
            start_total = time.time()
            # Using PlannerAgent from autogen_agents for multi-agent orchestration
            from autogen_agents.planner_agent import PlannerAgent
            agent = PlannerAgent(llm_model=llm_model, temperature=temperature, max_tokens=max_tokens)
            answer = agent.run_autonomous(user_input)
            total_time = time.time() - start_total
        answer += f"\n\n[Total Agent Orchestration Time: {total_time:.2f} sec]"
        st.session_state.conversation.append({"sender": "bot", "text": answer})
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

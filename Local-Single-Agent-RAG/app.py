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
import webbrowser
import tempfile

from utils.pdf_utils import pdf_to_markdown
from utils.embedding import get_embedding_for_text
from utils.local_llm import generate_response
from utils.chunking import chunk_text

# Set page config as the first Streamlit command!
st.set_page_config(page_title="Single Agent AI Bot (RAG)", layout="wide")

# Global configuration
LLM_PROXY_URL = "http://127.0.0.1:11434"
INPUT_FOLDER = "input"
INDEX_FILE = "index.json"
if not os.path.exists(INPUT_FOLDER):
    os.makedirs(INPUT_FOLDER)

# -------------------------------
# Retrieval Helper Functions
# -------------------------------
def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two numpy arrays."""
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    
    if vec1.shape != vec2.shape or vec1.size == 0 or vec2.size == 0:
        return 0  # Avoid errors if embeddings are incorrectly shaped

    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0


def retrieve_context(query):
    """
    Retrieves the most relevant chunk based on cosine similarity.
    """
    query_embedding = get_embedding_for_text(query)
    if not query_embedding or not os.path.exists(INDEX_FILE):
        return ""

    query_embedding = np.array(query_embedding)
    
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    best_similarity = -1
    best_text = ""
    
    for doc in index_data:
        for chunk in doc.get("chunks", []):
            emb = chunk.get("embedding", [])
            if emb:
                emb = np.array(emb)
                sim = cosine_similarity(query_embedding, emb)
                if sim > best_similarity:
                    best_similarity = sim
                    best_text = chunk.get("chunk_text", "")
    
    return best_text

def query_knowledge_graph(question, temperature, max_tokens, model):
    """
    Handles retrieval, embedding, and response generation timing.
    """
    start_total = time.time()

    # Step 1: Retrieve context
    start_retrieval = time.time()
    context = retrieve_context(question)
    retrieval_time = time.time() - start_retrieval

    if not context:
        context = "No relevant document context found."

    # Step 2: Construct Prompt
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer clearly."

    # Step 3: Embedding Time
    start_embedding = time.time()
    get_embedding_for_text(question)  # Just to measure time
    embedding_time = time.time() - start_embedding

    # Step 4: Generate Response
    start_response = time.time()
    response = generate_response(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    response_time = time.time() - start_response

    # Step 5: Total Processing Time
    total_time = time.time() - start_total

    st.write(f"⏳ **Timing Details**")
    st.write(f"- ⏱ Retrieval Time: `{retrieval_time:.2f} sec`")
    st.write(f"- ⏳ Embedding Time: `{embedding_time:.2f} sec`")
    st.write(f"- 🤖 LLM Response Time: `{response_time:.2f} sec`")
    st.write(f"- 🏁 Total Processing Time: `{total_time:.2f} sec`")

    return response


# -------------------------------
# Graph Visualization Functions
# -------------------------------

def build_graph(index_file=INDEX_FILE, threshold=0.7):
    """
    Builds a graph with documents and chunks as nodes.
    """
    G = nx.Graph()
    if not os.path.exists(index_file):
        return G
    with open(index_file, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    # Add document nodes
    for doc in index_data:
        G.add_node(doc["filename"], label=doc["filename"], group="document", size=20)  # Ensure group="document"

    # Add chunk nodes and connect them to documents
    chunk_nodes = []
    for doc in index_data:
        for i, chunk in enumerate(doc.get("chunks", [])):
            node_id = f"{doc['filename']}_chunk_{i}"
            G.add_node(
                node_id,
                label=f"Chunk {i}",
                title=f"Document: {doc['filename']}\nChunk: {chunk['chunk_text'][:100]}...",
                group="chunk",  # Assign group="chunk" explicitly
                size=10
            )
            G.add_edge(doc["filename"], node_id)  # Connect chunk to its document
            chunk_nodes.append((node_id, chunk.get("embedding", [])))

    # Add edges between chunks based on similarity
    for i in range(len(chunk_nodes)):
        for j in range(i + 1, len(chunk_nodes)):
            node1, emb1 = chunk_nodes[i]
            node2, emb2 = chunk_nodes[j]

            emb1 = np.array(emb1)
            emb2 = np.array(emb2)
            if emb1.size == 0 or emb2.size == 0:
                continue
            
            sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))
            if sim >= threshold:
                G.add_edge(node1, node2, weight=sim)
    
    return G



def visualize_graph(G):
    """
    Generates an interactive graph using PyVis.
    """
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")

    if len(G.nodes) == 0:
        st.sidebar.error("⚠️ Graph is empty! No nodes to visualize.")
        return None

    net.from_nx(G)

    for node in net.nodes:
        node["color"] = "lightblue" if node["group"] == "document" else "lightgreen"

    # Debugging: Check if PyVis can generate the HTML file
    try:
        temp_html_path = "graph.html"
        net.show(temp_html_path)

        # Ensure the file was written correctly
        if not os.path.exists(temp_html_path) or os.path.getsize(temp_html_path) == 0:
            st.sidebar.error("⚠️ Graph HTML file was not generated correctly.")
            return None

        with open(temp_html_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.sidebar.error(f"⚠️ Graph visualization error: {str(e)}")
        return None



# -------------------------------
# Sidebar: Settings and Graph Visualization
# -------------------------------
st.sidebar.header("Settings")
temperature = st.sidebar.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
max_tokens = st.sidebar.number_input("Max Tokens", min_value=50, max_value=1024, value=256, step=50)
llm_model = st.sidebar.selectbox("LLM Model", ["llama3.2:latest", "deepseek-r1:14b"])
if st.sidebar.button("Clear Conversation"):
    st.session_state.conversation = []
    st.rerun()

# if st.sidebar.button("Visualize Graph RAG"):
#     G = build_graph()
    
#     # Display node & edge count
#     st.sidebar.info(f"📊 Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

#     if G.number_of_nodes() == 0:
#         st.sidebar.warning("⚠️ No documents indexed to display.")
#     else:
#         graph_html = visualize_graph(G)
#         if graph_html:
#             components.html(graph_html, height=600, scrolling=True)
#         else:
#             st.sidebar.error("⚠️ Graph rendering failed.")


# -------------------------------
# Section 1: PDF Upload, Extraction & Indexing
# -------------------------------

color = "blue"
st.header("Upload PDF Document and Update Index")
uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_pdf is not None:
    start_time = time.time()
    file_path = os.path.join("input", uploaded_pdf.name)  # Replace "input" with your actual INPUT_FOLDER
    with open(file_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.success(f"Uploaded {uploaded_pdf.name} successfully!")
    
    markdown_content = pdf_to_markdown(file_path)
    if markdown_content:
        extraction_time = time.time() - start_time
        st.markdown(f'<p style="color:{color};">Extraction completed in {extraction_time:.2f} seconds.</p>', unsafe_allow_html=True)
        
        show_extracted_text = st.checkbox("Show Extracted Text")
        if show_extracted_text:
            st.write("Extracted Content:")
            st.markdown(markdown_content)
        
        # --- Indexing: split document into chunks and generate embeddings for each chunk ---
        chunks = chunk_text(markdown_content, method="fixed_window", max_words=1000, overlap=200)
        document_record = {
            "filename": uploaded_pdf.name,
            "text": markdown_content,
            "chunks": []
        }
        embed_start = time.time()
        for chunk in chunks:
            emb = get_embedding_for_text(chunk)
            document_record["chunks"].append({
                "chunk_text": chunk,
                "embedding": emb
            })
        embed_time = time.time() - embed_start
        st.markdown(f'<p style="color:{color};">Embedding generation completed in {embed_time:.2f} seconds.</p>', unsafe_allow_html=True)
        
        # Load existing index data if available.
        if os.path.exists("index.json"): #replace "index.json" with your INDEX_FILE
            with open("index.json", "r", encoding="utf-8") as f:
                index_data = json.load(f)
        else:
            index_data = []
        index_data.append(document_record)
        with open("index.json", "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        st.success(f"Index updated with {uploaded_pdf.name}.")
    else:
        st.error("Failed to extract content from the PDF.")

# -------------------------------
# Section 2: Chat Interface
# -------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []

st.title("Single Agent AI Bot (RAG)")
st.markdown("This application retrieves context from indexed documents and uses a local LLM to generate responses.")

with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for entry in st.session_state.conversation:
        if entry["sender"] == "user":
            st.markdown(f'<div class="user-msg"><strong>User:</strong> {entry["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg"><strong>Bot:</strong> {entry["text"]}</div>', unsafe_allow_html=True)
    
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Enter your question", "")
        submit_button = st.form_submit_button("Send")
    
    if submit_button and user_input:
        st.session_state.conversation.append({"sender": "user", "text": user_input})
        with st.spinner("Generating answer..."):
            answer = query_knowledge_graph(
                question=user_input,
                temperature=temperature,
                max_tokens=max_tokens,
                model=llm_model,
            )
        st.session_state.conversation.append({"sender": "bot", "text": answer})
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
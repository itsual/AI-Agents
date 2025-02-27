import requests
import streamlit as st
import numpy as np

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    if a.size == 0 or b.size == 0:
        return 0
    dot = np.dot(a, b)
    normA = np.linalg.norm(a)
    normB = np.linalg.norm(b)
    if normA == 0 or normB == 0:
        return 0
    return dot / (normA * normB)

LLM_PROXY_URL = "http://127.0.0.1:11434"
EMBEDDING_ENDPOINT = f"{LLM_PROXY_URL}/api/embeddings"

def get_embedding_for_text(text, model="nomic-embed-text:latest", timeout=15):
    """
    Sends a request to the local embedding API to generate an embedding
    for the provided text using the specified model.
    """
    payload = {
        "model": model,
        "prompt": text
    }
    try:
        resp = requests.post(EMBEDDING_ENDPOINT, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("embedding", [])
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return []

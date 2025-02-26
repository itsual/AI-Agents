import requests
import streamlit as st

LLM_PROXY_URL = "http://127.0.0.1:11434"
EMBEDDING_ENDPOINT = f"{LLM_PROXY_URL}/api/embeddings"

def get_embedding_for_text(text, model="nomic-embed-text:latest", timeout=15):
    """
    Sends a request to the local embedding API (nomic-embed-text) to generate an embedding
    for the provided text.
    """
    payload = {
        "model": model,
        "prompt": text
    }
    try:
        response = requests.post(EMBEDDING_ENDPOINT, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("embedding", [])
    except Exception as e:
        st.error(f"Error retrieving embedding: {e}")
        try:
            st.error(f"Response text: {response.text}")
        except Exception:
            st.error("No response available.")
        return []

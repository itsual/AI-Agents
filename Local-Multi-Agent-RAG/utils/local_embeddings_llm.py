import requests
import streamlit as st

LLM_PROXY_URL = "http://127.0.0.1:11434"
COMPLETIONS_ENDPOINT = f"{LLM_PROXY_URL}/v1/completions"

def generate_response(prompt, model, temperature=0.7, max_tokens=256):
    payload = {
        "prompt": prompt,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        r = requests.post(COMPLETIONS_ENDPOINT, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0].get("text", "").strip()
        else:
            return "No response generated."
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Error generating response."

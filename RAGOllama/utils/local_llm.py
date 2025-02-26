import requests

LLM_PROXY_URL = "http://127.0.0.1:11434"
COMPLETIONS_ENDPOINT = f"{LLM_PROXY_URL}/v1/completions"

def generate_response(prompt, model, temperature=0.7, max_tokens=256):
    """
    Sends a request to the local LLM completions API to generate a response for the given prompt.

    Parameters:
        prompt (str): The text prompt to send to the model.
        model (str): The selected LLM model, dynamically passed from the app.
        temperature (float): The randomness of the response. Higher values increase diversity.
        max_tokens (int): The maximum length of the generated response.

    Returns:
        str: The generated response text.
    """
    payload = {
        "prompt": prompt,
        "model": model,  # Dynamically use the selected model
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(COMPLETIONS_ENDPOINT, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0].get("text", "").strip()
        else:
            return "No response generated."
    except Exception as e:
        print(f"Error calling LLM: {e}")
        try:
            print("Response text:", response.text)
        except Exception:
            print("No response available.")
        return "Error generating response."

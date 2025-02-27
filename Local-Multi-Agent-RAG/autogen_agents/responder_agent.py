import time
from utils.local_embeddings_llm import generate_response

class ResponderAgent:
    """
    Agent that generates a final response using the local LLM.
    """
    def __init__(self, model, temperature=0.7, max_tokens=256):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self, prompt):
        start_t = time.time()
        answer = generate_response(prompt, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)
        gen_time = time.time()-start_t
        return answer, gen_time
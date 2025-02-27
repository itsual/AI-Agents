import time
from autogen_agents.retrieval_agent import RetrievalAgent
from autogen_agents.responder_agent import ResponderAgent

class PlannerAgent:
    """
    High-level orchestrator that uses a retrieval agent and a responder agent.
    """
    def __init__(self, llm_model, temperature=0.7, max_tokens=256):
        self.retrieval_agent = RetrievalAgent(index_file="index.json")
        self.responder_agent = ResponderAgent(model=llm_model, temperature=temperature, max_tokens=max_tokens)


    def run_autonomous(self, user_query):
        retrieval_start = time.time()
        context = self.retrieval_agent.retrieve_context(user_query)
        retrieval_time = time.time() - retrieval_start
        if not context:
            context = "No relevant context found."

        prompt = (
            f"Using the following context:\n{context}\n\n"
            f"Answer the question:\n{user_query}\n\n"
            "Provide a clear, concise answer."
        )
        answer, gen_time = self.responder_agent.generate(prompt)
        result = (
            f"{answer}\n\n[Retrieval Time: {retrieval_time:.2f}s, Generation Time: {gen_time:.2f}s]"
        )
        return result

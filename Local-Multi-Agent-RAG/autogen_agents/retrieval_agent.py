# autogen_agents/retrieval_agent.py

import os
import json
import numpy as np
from utils.embedding import get_embedding_for_text, cosine_similarity

def adjust_embedding(embedding, target_shape):
    """
    Adjusts the embedding to match the target shape.
    If the current embedding is shorter, pads with zeros.
    If it is longer, truncates to the target length.
    """
    embedding = np.array(embedding)
    current_dim = embedding.shape[0]
    target_dim = target_shape[0]
    if current_dim < target_dim:
        padded = np.zeros(target_shape)
        padded[:current_dim] = embedding
        return padded
    elif current_dim > target_dim:
        return embedding[:target_dim]
    else:
        return embedding

class RetrievalAgent:
    """
    A specialized agent that retrieves relevant context from an index.
    Called by the PlannerAgent.
    """
    def __init__(self, index_file="index.json"):
        self.index_file = index_file

    def retrieve_context(self, query):
        if not os.path.exists(self.index_file):
            return ""
        query_emb = get_embedding_for_text(query)
        if not query_emb:
            return ""
        # Ensure query embedding is a NumPy array.
        query_emb = np.array(query_emb)
        
        with open(self.index_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        best_sim = -1
        best_chunk = ""
        for doc in index_data:
            for chunk in doc.get("chunks", []):
                emb = chunk.get("embedding", [])
                if emb:
                    emb = np.array(emb)
                    # Adjust chunk embedding to match the query embedding shape if needed.
                    if emb.shape != query_emb.shape:
                        emb = adjust_embedding(emb, query_emb.shape)
                    sim = cosine_similarity(query_emb, emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_chunk = chunk.get("chunk_text", "")
        return best_chunk

# Multi-Agent AI RAG Bot

A local-first, multi-agent AI system that uses Retrieval-Augmented Generation (RAG) techniques to answer questions based on your own documents. This repository demonstrates how multiple specialized agents can work together—one for retrieval, another for response generation, a planner to orchestrate them, and an optional agent for semantic chunking.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Models & Dependencies](#models--dependencies)
- [Ollama Usage](#ollama-usage)
- [How It Works](#how-it-works)
- [License](#license)

---

## Overview

In typical RAG systems, you have a single pipeline:
1. Ingest documents
2. Embed them
3. Retrieve relevant chunks
4. Generate a final answer

Here, we go a step further and assign each responsibility to a specialized agent. These agents are coordinated by a **PlannerAgent**, creating a more modular, extensible system that can be adapted or expanded for additional tasks like semantic chunking, document classification, or advanced analytics.

---

## Features

1. **Multi-Agent Orchestration**  
   - **PlannerAgent**: High-level controller that orchestrates other agents.  
   - **RetrievalAgent**: Fetches the most relevant chunks from a local index.  
   - **ResponderAgent**: Crafts the final response via an LLM.  
   - *(Optional)* **SemanticChunkAgent**: Performs semantic chunking to improve chunk coherence.

2. **Local Embeddings & LLM**  
   - The system uses local embeddings for document indexing (e.g., [nomic-embed-text](https://github.com/nomic-ai/nomic) or others).
   - Local LLM inference via [Ollama](https://github.com/jmorganca/ollama) to ensure data privacy.

3. **Flexible Document Ingestion**  
   - Supports uploading PDF files (converted to Markdown).
   - Optionally allows for advanced chunking (semantic or fixed-window).

4. **RAG Graph Visualization**  
   - You can visualize your indexed documents as a graph. This helps to see how documents (and optionally their chunks) are connected by semantic similarity.

5. **Agentic Framework**  
   - Each agent is purpose-built for a specific task. The code is cleanly separated, allowing easy replacement or extension of any agent.

---

## Architecture


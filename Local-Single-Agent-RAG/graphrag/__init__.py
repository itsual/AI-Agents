#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
GraphRAG Package

This package handles the initialization and indexing of documents to build a knowledge graph.
It provides functionality to create the input folder, copy configuration settings, and generate
embeddings for your documents.
"""

from .index import init_index, index_documents

__all__ = [
    "init_index",
    "index_documents",
]


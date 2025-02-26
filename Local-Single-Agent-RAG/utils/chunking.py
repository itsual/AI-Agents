import re

def chunk_text(text, method="fixed_window", **kwargs):
    """
    Splits text into chunks based on the specified method.
    """
    if method == "fixed_window":
        return chunk_text_fixed_window(text, **kwargs)
    elif method == "semantic":
        return chunk_text_semantic(text, **kwargs)
    else:
        raise ValueError(f"Invalid chunking method: {method}")

def chunk_text_fixed_window(text, max_words=1000, overlap=200):
    """
    Splits text into chunks of at most max_words words, with an overlap between chunks.
    """
    words = text.split()
    chunks = []  # Initialize empty list
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks

def chunk_text_semantic(text):
    """
    Splits text into chunks based on semantic units like paragraphs.
    """
    chunks = text.split('\n\n')
    chunks = [c.split('\n') for c in chunks]
    chunks = [item for sublist in chunks for item in sublist]
    return chunks

from langchain.text_splitter import MarkdownTextSplitter

def chunk_text_markdown(text, chunk_size=400, chunk_overlap=0):
    """
    Splits Markdown text into chunks using MarkdownTextSplitter.
    """
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks

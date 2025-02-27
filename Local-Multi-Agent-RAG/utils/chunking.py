def chunk_text(text, method="fixed_window", max_words=1000, overlap=200):
    """
    Splits text into chunks.
    Default method is fixed_window.
    """
    if method == "fixed_window":
        return chunk_text_fixed_window(text, max_words, overlap)
    else:
        return chunk_text_semantic(text)

def chunk_text_fixed_window(text, max_words=1000, overlap=200):
    words = text.split()
    chunks = []
    step = max_words - overlap
    if step < 1:
        step = 1
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def chunk_text_semantic(text):
    paragraphs = text.split("\n\n")
    chunks = [p.strip() for p in paragraphs if p.strip()]
    return chunks

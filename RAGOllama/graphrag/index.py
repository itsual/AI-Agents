import json
import os
from utils.chunking import chunk_text
from utils.embedding import get_embedding_for_text

INDEX_FILE = "index.json"
INPUT_FOLDER = "input"

# Ensure input folder exists
if not os.path.exists(INPUT_FOLDER):
    os.makedirs(INPUT_FOLDER)

# Load existing index data if available
if os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        index_data = json.load(f)
else:
    index_data = []

documents = []  # List to store document records

def process_document(filename, content):
    """
    Processes a document: splits into chunks, embeds, and updates the index.
    """
    doc_record = {
        "filename": filename,
        "text": content,
        "chunks": []  # Ensure chunks are initialized
    }

    chunks = chunk_text(content, method="fixed_window", max_words=1000, overlap=200)

    for chunk in chunks:
        embedding = get_embedding_for_text(chunk)
        doc_record["chunks"].append({
            "chunk_text": chunk,
            "embedding": embedding
        })

    return doc_record

# Iterate through documents and update the index
for doc in documents:
    processed_doc = process_document(doc["filename"], doc["text"])
    index_data.append(processed_doc)

# Save the updated index
with open(INDEX_FILE, "w", encoding="utf-8") as f:
    json.dump(index_data, f, ensure_ascii=False, indent=2)

print("Index updated successfully!")

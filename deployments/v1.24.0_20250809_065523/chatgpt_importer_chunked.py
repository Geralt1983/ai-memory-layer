# chatgpt_importer_chunked.py
import json
import os
from typing import List, Dict
from integrations.embeddings import OpenAIEmbeddings
from storage.faiss_store import FaissVectorStore


CHUNK_LIMIT = 100  # max lines per diff block


def chunk_diff_text(diff_text: str, max_lines: int = CHUNK_LIMIT) -> List[str]:
    lines = diff_text.splitlines()
    chunks = []
    for i in range(0, len(lines), max_lines):
        chunk = lines[i:i + max_lines]
        if chunk:
            chunks.append("\n".join(chunk))
    return chunks


def load_commit_json(json_path: str) -> List[Dict]:
    with open(json_path, 'r') as f:
        return json.load(f)


def process_commit(json_data: List[Dict], embedder, index_path: str):
    texts = []
    metadatas = []
    
    for entry in json_data:
        commit = entry.get("hash")
        title = entry.get("title")
        date = entry.get("date")
        diff_text = entry.get("diff", "")

        for chunk in chunk_diff_text(diff_text):
            texts.append(chunk)
            metadatas.append({
                "commit": commit,
                "title": title,
                "date": date,
                "length": len(chunk),
            })

    print(f"🔢 Chunks to embed: {len(texts)}")
    embeddings = embedder.embed_documents(texts)
    store = FaissVectorStore()
    store.add_documents(texts, embeddings, metadatas)
    store.save(index_path)
    print(f"✅ Saved FAISS index to: {index_path}")


if __name__ == "__main__":
    import sys

    json_path = sys.argv[1]  # path to commit .json
    index_out = sys.argv[2]  # output FAISS base path (no extension)
    
    embedder = OpenAIEmbeddings()
    json_data = load_commit_json(json_path)
    process_commit(json_data, embedder, index_out)
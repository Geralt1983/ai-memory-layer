#!/usr/bin/env python3
"""
Rebuild FAISS index using cleaned memories
"""

import sys
sys.path.append('.')

# Update the rebuild script to use cleaned data
from rebuild_faiss_index import rebuild_faiss_index

# Override the global variables  
import rebuild_faiss_index
rebuild_faiss_index.MEMORY_FILE = "./data/chatgpt_memories_cleaned.json"
rebuild_faiss_index.FAISS_INDEX_PATH = "./data/faiss_chatgpt_cleaned"

print("🧹 Rebuilding FAISS index with CLEANED memories...")
print("=" * 50)

# Run the rebuild
rebuild_faiss_index.rebuild_faiss_index()

print("\n✨ Cleaned FAISS index ready at:")
print("   - data/faiss_chatgpt_cleaned.index")
print("   - data/faiss_chatgpt_cleaned.pkl")
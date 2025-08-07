#!/usr/bin/env python3
"""
Batch load memories to avoid timeout issues
"""
import json
import asyncio
import aiohttp
import sys
from pathlib import Path

async def load_memories_batch(api_url: str, memories: list, batch_size: int = 100):
    """Load memories in batches"""
    async with aiohttp.ClientSession() as session:
        total = len(memories)
        for i in range(0, total, batch_size):
            batch = memories[i:i + batch_size]
            print(f"Loading batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch)} memories)...")
            
            for memory in batch:
                try:
                    async with session.post(
                        f"{api_url}/memories",
                        json={
                            "content": memory.get("content", ""),
                            "metadata": memory.get("metadata", {})
                        }
                    ) as response:
                        if response.status != 201:
                            print(f"Failed to add memory: {response.status}")
                except Exception as e:
                    print(f"Error adding memory: {e}")
            
            # Small delay between batches
            await asyncio.sleep(1)
            print(f"Completed batch {i//batch_size + 1}")

async def main():
    # Load memories from backup
    memory_file = Path(sys.argv[1] if len(sys.argv) > 1 else "data/chatgpt_memories.json.backup")
    
    if not memory_file.exists():
        print(f"Memory file not found: {memory_file}")
        return
    
    print(f"Loading memories from {memory_file}...")
    with open(memory_file, 'r') as f:
        memories = json.load(f)
    
    print(f"Found {len(memories)} memories to load")
    
    # Load in batches
    api_url = "http://localhost:8000"
    await load_memories_batch(api_url, memories, batch_size=50)
    
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
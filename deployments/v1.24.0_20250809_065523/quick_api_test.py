#!/usr/bin/env python3
"""Quick API test without loading all memories"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "AI Memory Layer"}

@app.get("/memories/stats")
async def memory_stats():
    return {
        "total_memories": 23710,
        "memory_types": {
            "conversation": 23710,
            "commit": 0,
            "summary": 0
        },
        "recent_count": 100,
        "indexed": True
    }

@app.get("/stats")
async def stats():
    return {
        "memories": 23710,
        "conversations": 1,
        "version": "1.14.0",
        "status": "operational"
    }

if __name__ == "__main__":
    print("Starting quick test API on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
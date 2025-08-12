#!/usr/bin/env python3
"""
Start the memory API server
"""

import sys
import os
from pathlib import Path

# Add lib directory to path
lib_dir = Path(__file__).parent / "lib"
sys.path.insert(0, str(lib_dir))

# Start the memory API
if __name__ == "__main__":
    import uvicorn
    
    print("🧠 Starting Memory API server on port 8001...")
    uvicorn.run("lib.memory_api:app", host="0.0.0.0", port=8001, reload=False)
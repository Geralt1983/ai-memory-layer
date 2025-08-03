#!/usr/bin/env python3
"""
Simple test script that doesn't require numpy/heavy dependencies
Tests core Memory functionality with mock data
"""

from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json

@dataclass
class SimpleMemory:
    """Simplified Memory class without numpy dependencies"""
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'embedding': self.embedding,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }

class SimpleMemoryEngine:
    """Simplified MemoryEngine for testing without external dependencies"""
    
    def __init__(self):
        self.memories: List[SimpleMemory] = []
    
    def add_memory(self, memory: SimpleMemory):
        self.memories.append(memory)
    
    def get_recent_memories(self, k: int = 5) -> List[SimpleMemory]:
        return sorted(self.memories, key=lambda m: m.timestamp, reverse=True)[:k]
    
    def get_memory_count(self) -> int:
        return len(self.memories)

def test_simple_functionality():
    """Test basic memory functionality"""
    print("Testing Simple Memory Engine...")
    
    # Create engine
    engine = SimpleMemoryEngine()
    print(f"✓ Engine created, memory count: {engine.get_memory_count()}")
    
    # Create test memories
    memories = [
        SimpleMemory(
            content="This is a test memory about AI",
            embedding=[0.1, 0.2, 0.3],
            metadata={"type": "test", "category": "ai"},
            timestamp=datetime.now()
        ),
        SimpleMemory(
            content="Another memory about memory management",
            embedding=[0.4, 0.5, 0.6],
            metadata={"type": "test", "category": "memory"},
            timestamp=datetime.now()
        )
    ]
    
    # Add memories
    for memory in memories:
        engine.add_memory(memory)
        print(f"✓ Added memory: {memory.content[:30]}...")
    
    # Test retrieval
    recent = engine.get_recent_memories(10)
    print(f"✓ Retrieved {len(recent)} recent memories")
    
    # Test memory data
    for i, memory in enumerate(recent):
        print(f"  {i+1}. {memory.content[:50]}...")
        print(f"     Metadata: {memory.metadata}")
    
    print(f"\n✓ Test completed successfully!")
    print(f"✓ Total memories stored: {engine.get_memory_count()}")

if __name__ == "__main__":
    test_simple_functionality()
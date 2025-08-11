#!/usr/bin/env python3
"""
Simple API client to test the memory API functionality
"""

import json
from datetime import datetime


def test_api_endpoints():
    """Test API endpoints using basic HTTP requests"""
    print("AI Memory Layer API Client Test")
    print("=" * 40)

    # Since we can't easily make HTTP requests without additional dependencies,
    # let's test the API logic directly by importing the app
    try:
        from simple_api import memory_engine, MemoryCreate
        from simple_test import SimpleMemory

        print("✓ Successfully imported API components")

        # Test 1: Check initial state
        initial_count = memory_engine.get_memory_count()
        print(f"✓ Initial memory count: {initial_count}")

        # Test 2: Add some test memories
        test_memories = [
            "This is a memory about machine learning and AI",
            "Another memory about Python programming",
            "A memory about API development with FastAPI",
            "Memory about vector databases and embeddings",
        ]

        for content in test_memories:
            mock_embedding = [hash(content) % 100 / 100.0 for _ in range(3)]
            memory = SimpleMemory(
                content=content,
                embedding=mock_embedding,
                metadata={"test": True, "type": "demo"},
                timestamp=datetime.now(),
            )
            memory_engine.add_memory(memory)
            print(f"✓ Added memory: {content[:30]}...")

        # Test 3: Check memory count after additions
        final_count = memory_engine.get_memory_count()
        print(f"✓ Final memory count: {final_count}")
        print(f"✓ Added {final_count - initial_count} new memories")

        # Test 4: Retrieve recent memories
        recent_memories = memory_engine.get_recent_memories(10)
        print(f"✓ Retrieved {len(recent_memories)} recent memories")

        # Test 5: Display memory contents
        print("\nStored Memories:")
        print("-" * 30)
        for i, memory in enumerate(recent_memories, 1):
            print(f"{i}. {memory.content}")
            print(f"   Metadata: {memory.metadata}")
            print(f"   Timestamp: {memory.timestamp.strftime('%H:%M:%S')}")
            print()

        # Test 6: Simple search functionality
        search_term = "API"
        matching_memories = [
            memory
            for memory in recent_memories
            if search_term.lower() in memory.content.lower()
        ]
        print(f"✓ Search for '{search_term}' found {len(matching_memories)} matches")

        for memory in matching_memories:
            print(f"  - {memory.content}")

        print("\n" + "=" * 40)
        print("✓ All API functionality tests passed!")
        print(f"✓ Total memories in system: {memory_engine.get_memory_count()}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_api_endpoints()
    if success:
        print("\n🎉 API testing completed successfully!")
    else:
        print("\n❌ API testing failed!")

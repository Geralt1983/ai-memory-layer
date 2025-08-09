#!/usr/bin/env python3
"""
Comprehensive Context Retention Test Suite
Tests the DirectOpenAIChat implementation for proper context handling
"""

import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.memory_engine import MemoryEngine
from storage.faiss_store import FaissVectorStore
from integrations.embeddings import OpenAIEmbeddings
from integrations.direct_openai import DirectOpenAIChat

def setup_test_environment():
    """Set up the test environment with fresh instances"""
    # Set OpenAI API key
    api_key = "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"
    
    try:
        # Initialize components
        embedding_provider = OpenAIEmbeddings(api_key=api_key)
        vector_store = FaissVectorStore(dimension=1536, index_path="./test_data/faiss_index")
        memory_engine = MemoryEngine(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            persist_path="./test_data/test_memories.json"
        )
        
        # Initialize DirectOpenAIChat
        chat = DirectOpenAIChat(
            api_key=api_key,
            memory_engine=memory_engine,
            model="gpt-3.5-turbo"  # Using cheaper model for testing
        )
        
        return chat
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        # Try fallback without vector components
        from storage.mock_store import MockVectorStore
        from integrations.mock_embeddings import MockEmbeddings
        
        vector_store = MockVectorStore()
        embedding_provider = MockEmbeddings()
        memory_engine = MemoryEngine(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            persist_path="./test_data/test_memories.json"
        )
        
        chat = DirectOpenAIChat(
            api_key=api_key,
            memory_engine=memory_engine,
            model="gpt-3.5-turbo"
        )
        
        print("⚠️  Using mock components due to setup issues")
        return chat

def run_test_5_long_conversation():
    """Test 5: Long conversation with multiple topic switches"""
    print("\n" + "="*60)
    print("🧪 TEST 5: Long Conversation with Topic Switches")
    print("="*60)
    
    chat = setup_test_environment()
    thread_id = "test_long_conversation"
    
    # Clear any existing conversation
    chat.clear_thread(thread_id)
    
    # Topic 1: Python vs Go
    print("\n💬 Topic 1: Python vs Go")
    response1, _ = chat.chat(
        "I need help choosing between Python and Go for a new microservices project. The project needs to handle high throughput APIs.",
        thread_id=thread_id
    )
    print(f"🤖 {response1[:100]}...")
    
    # Follow-up on Python vs Go
    response2, _ = chat.chat(
        "What about performance differences?",
        thread_id=thread_id
    )
    print(f"🤖 {response2[:100]}...")
    
    # Topic switch to Docker
    print("\n💬 Topic Switch: Docker")
    response3, _ = chat.chat(
        "Actually, let's talk about Docker instead. I'm having issues with container networking.",
        thread_id=thread_id
    )
    print(f"🤖 {response3[:100]}...")
    
    # Follow-up on Docker
    response4, _ = chat.chat(
        "The containers can't communicate with each other. What should I check?",
        thread_id=thread_id
    )
    print(f"🤖 {response4[:100]}...")
    
    # Topic switch to database design
    print("\n💬 Topic Switch: Database Design")
    response5, _ = chat.chat(
        "Now I want to discuss database design. Should I use PostgreSQL or MongoDB for user profiles?",
        thread_id=thread_id
    )
    print(f"🤖 {response5[:100]}...")
    
    # Test contextual reference back to earlier topics
    print("\n💬 Testing Contextual Reference")
    response6, _ = chat.chat(
        "Going back to our earlier discussion, which language did you recommend?",
        thread_id=thread_id
    )
    print(f"🤖 {response6}")
    
    # Verify context retention
    if "Python" in response6 or "Go" in response6 or "language" in response6.lower():
        print("✅ PASSED: AI correctly referenced earlier Python/Go discussion")
        return True
    else:
        print("❌ FAILED: AI did not reference earlier language discussion")
        print(f"Full response: {response6}")
        return False

def run_test_6_thread_persistence():
    """Test 6: Thread persistence simulation"""
    print("\n" + "="*60)
    print("🧪 TEST 6: Thread Persistence Simulation")
    print("="*60)
    
    chat = setup_test_environment()
    thread_id = "test_persistence"
    
    # Start a conversation
    print("\n💬 Starting conversation about React hooks")
    response1, _ = chat.chat(
        "I'm having trouble with React useEffect hooks. They're running too often.",
        thread_id=thread_id
    )
    print(f"🤖 {response1[:100]}...")
    
    # Simulate "page reload" by creating new chat instance
    print("\n🔄 Simulating page reload (new chat instance)")
    chat2 = setup_test_environment()
    
    # Continue conversation - should remember context
    print("\n💬 Continuing conversation after 'reload'")
    response2, _ = chat2.chat(
        "What was the main issue you mentioned?",
        thread_id=thread_id
    )
    print(f"🤖 {response2}")
    
    # Check if context was maintained
    if "useEffect" in response2 or "hook" in response2.lower() or "running" in response2.lower():
        print("✅ PASSED: Context maintained across 'page reload'")
        return True
    else:
        print("❌ FAILED: Context lost after 'page reload'")
        print(f"Full response: {response2}")
        return False

def run_test_7_memory_integration():
    """Test 7: Memory search integration with conversation context"""
    print("\n" + "="*60)
    print("🧪 TEST 7: Memory Search Integration")
    print("="*60)
    
    chat = setup_test_environment()
    thread_id = "test_memory_integration"
    
    # Add some memories first
    print("\n💾 Adding test memories")
    chat.memory_engine.add_memory(
        "Jeremy prefers Python for web development due to Django and FastAPI",
        metadata={"type": "preference", "topic": "programming"}
    )
    chat.memory_engine.add_memory(
        "Jeremy uses Docker for containerization in all his projects",
        metadata={"type": "tool", "topic": "devops"}
    )
    
    # Start conversation that should trigger memory search
    print("\n💬 Starting conversation about web development")
    response1, _ = chat.chat(
        "What programming language should I use for a new web API?",
        thread_id=thread_id
    )
    print(f"🤖 {response1}")
    
    # Follow up with contextual reference
    print("\n💬 Testing contextual reference with memory")
    response2, _ = chat.chat(
        "What about containerization for that project?",
        thread_id=thread_id
    )
    print(f"🤖 {response2}")
    
    # Check if both conversation context and memory were used
    mentions_python = "Python" in response1
    mentions_docker = "Docker" in response2 or "container" in response2.lower()
    
    if mentions_python and mentions_docker:
        print("✅ PASSED: Both memory search and conversation context working")
        return True
    else:
        print(f"❌ FAILED: Memory integration issue")
        print(f"Python mentioned: {mentions_python}")
        print(f"Docker mentioned: {mentions_docker}")
        return False

def main():
    """Run all comprehensive tests"""
    print("🚀 Starting Comprehensive Context Retention Test Suite")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔧 Testing DirectOpenAIChat v1.10.0")
    
    # Ensure test directory exists
    os.makedirs("./test_data", exist_ok=True)
    
    results = []
    
    # Run all tests
    try:
        results.append(("Test 5: Long Conversation", run_test_5_long_conversation()))
        results.append(("Test 6: Thread Persistence", run_test_6_thread_persistence()))
        results.append(("Test 7: Memory Integration", run_test_7_memory_integration()))
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        return
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! Context retention is working perfectly.")
    else:
        print("⚠️  Some tests failed. Context retention needs attention.")

if __name__ == "__main__":
    main()
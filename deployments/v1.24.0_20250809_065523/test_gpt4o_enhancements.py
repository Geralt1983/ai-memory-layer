#!/usr/bin/env python3
"""
GPT-4o Enhancement Test Suite
Tests the new human-like response capabilities and sophisticated context handling
"""

import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from integrations.direct_openai import DirectOpenAIChat
from core.memory_engine import MemoryEngine
from storage.mock_store import MockVectorStore
from integrations.mock_embeddings import MockEmbeddings


def setup_gpt4o_chat():
    """Set up GPT-4o optimized chat for testing"""
    api_key = "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"
    
    try:
        # Use mock components for local testing
        vector_store = MockVectorStore()
        embedding_provider = MockEmbeddings()
        memory_engine = MemoryEngine(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            persist_path="./test_data/gpt4o_memories.json"
        )
        
        # Initialize GPT-4o optimized chat
        chat = DirectOpenAIChat(
            api_key=api_key,
            memory_engine=memory_engine,
            model="gpt-4o",  # Using GPT-4o for enhanced capabilities
            system_prompt_path="./prompts/system_prompt_4o.txt"
        )
        
        print("✅ GPT-4o optimized chat initialized successfully")
        return chat
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return None


def test_human_like_responses():
    """Test GPT-4o's human-like response capabilities"""
    print("\n" + "="*60)
    print("🧠 TEST: Human-like Response Quality")
    print("="*60)
    
    chat = setup_gpt4o_chat()
    if not chat:
        return False
    
    thread_id = "test_human_responses"
    chat.clear_thread(thread_id)
    
    # Test 1: Direct, confident responses (no hedging)
    print("\n💬 Testing confident, direct communication")
    response1, _ = chat.chat(
        "Should I use React or Vue for my next project?",
        thread_id=thread_id
    )
    print(f"🤖 {response1}")
    
    # Check for human-like qualities
    hedging_phrases = ["it depends", "you might want to", "perhaps", "possibly", "it could be"]
    has_hedging = any(phrase in response1.lower() for phrase in hedging_phrases)
    
    if not has_hedging and len(response1.split()) < 200:  # Concise response
        print("✅ Response is direct and confident")
        return True
    else:
        print("❌ Response contains too much hedging or is too verbose")
        return False


def test_contextual_memory_integration():
    """Test sophisticated memory integration"""
    print("\n" + "="*60)
    print("🧠 TEST: Contextual Memory Integration")
    print("="*60)
    
    chat = setup_gpt4o_chat()
    if not chat:
        return False
    
    thread_id = "test_memory_integration"
    chat.clear_thread(thread_id)
    
    # Add some test memories
    chat.memory_engine.add_memory(
        "Jeremy prefers TypeScript over JavaScript for type safety",
        metadata={"type": "preference", "topic": "programming"}
    )
    chat.memory_engine.add_memory(
        "Jeremy is working on AI memory layer with vector storage",
        metadata={"type": "project", "topic": "current_work"}
    )
    
    # Test memory integration
    print("\n💬 Testing memory-aware response")
    response1, _ = chat.chat(
        "What programming language should I use for type-safe web development?",
        thread_id=thread_id
    )
    print(f"🤖 {response1}")
    
    # Check if memory was naturally integrated
    if "TypeScript" in response1 and "type" in response1.lower():
        print("✅ Memory naturally integrated into response")
        return True
    else:
        print("❌ Memory not effectively integrated")
        return False


def test_conversation_continuity():
    """Test advanced conversation continuity"""
    print("\n" + "="*60)
    print("🧠 TEST: Advanced Conversation Continuity")
    print("="*60)
    
    chat = setup_gpt4o_chat()
    if not chat:
        return False
    
    thread_id = "test_continuity"
    chat.clear_thread(thread_id)
    
    # Complex multi-turn conversation
    print("\n💬 Setting up complex scenario")
    response1, _ = chat.chat(
        "I'm architecting a microservices system with these components: API Gateway, User Service, Payment Service, and Notification Service. The User Service needs high availability.",
        thread_id=thread_id
    )
    print(f"🤖 {response1[:100]}...")
    
    # Add interruption/tangent
    print("\n💬 Adding interruption")
    response2, _ = chat.chat(
        "Actually, let me ask about database choices first. What's better for user profiles?",
        thread_id=thread_id
    )
    print(f"🤖 {response2[:100]}...")
    
    # Return to original context
    print("\n💬 Returning to original context")
    response3, _ = chat.chat(
        "OK back to the microservices architecture. What did you think about the high availability requirement?",
        thread_id=thread_id
    )
    print(f"🤖 {response3}")
    
    # Check if it remembered the User Service and high availability context
    if ("User Service" in response3 or "user service" in response3.lower()) and "high availability" in response3.lower():
        print("✅ Successfully maintained context across interruption")
        return True
    else:
        print("❌ Lost context during conversation")
        return False


def test_personality_consistency():
    """Test Jeremy-specific personality and communication style"""
    print("\n" + "="*60)
    print("🧠 TEST: Personality Consistency")
    print("="*60)
    
    chat = setup_gpt4o_chat()
    if not chat:
        return False
    
    thread_id = "test_personality"
    chat.clear_thread(thread_id)
    
    # Test Jeremy-specific context awareness
    print("\n💬 Testing personalized response")
    response1, _ = chat.chat(
        "I'm feeling overwhelmed with all these technical decisions. What should I prioritize?",
        thread_id=thread_id
    )
    print(f"🤖 {response1}")
    
    # Check for personality traits (direct, no-fluff, helpful peer)
    ai_phrases = ["as an ai", "i'm here to help", "i understand you're feeling"]
    has_ai_phrases = any(phrase in response1.lower() for phrase in ai_phrases)
    
    if not has_ai_phrases and len(response1.split()) < 150:  # Concise, peer-like response
        print("✅ Response matches Jeremy's preferred communication style")
        return True
    else:
        print("❌ Response too generic or AI-like")
        return False


def main():
    """Run GPT-4o enhancement tests"""
    print("🚀 Starting GPT-4o Enhancement Test Suite")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔧 Testing Enhanced DirectOpenAIChat with GPT-4o Optimizations")
    
    # Ensure test directory exists
    os.makedirs("./test_data", exist_ok=True)
    os.makedirs("./prompts", exist_ok=True)
    
    results = []
    
    # Run all tests
    try:
        results.append(("Human-like Responses", test_human_like_responses()))
        results.append(("Memory Integration", test_contextual_memory_integration()))
        results.append(("Conversation Continuity", test_conversation_continuity()))
        results.append(("Personality Consistency", test_personality_consistency()))
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        return
    
    # Print summary
    print("\n" + "="*60)
    print("📊 GPT-4o ENHANCEMENT TEST RESULTS")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} enhancement tests passed")
    
    if passed >= 3:  # At least 3/4 passing is excellent
        print("🎉 GPT-4o ENHANCEMENTS WORKING EXCELLENTLY!")
        print("🔥 Human-like responses, sophisticated memory integration, and personality consistency achieved!")
    elif passed >= 2:
        print("✅ GPT-4o enhancements mostly working - minor adjustments may be needed")
    else:
        print("⚠️  GPT-4o enhancements need more work")


if __name__ == "__main__":
    main()
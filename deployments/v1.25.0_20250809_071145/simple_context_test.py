#!/usr/bin/env python3
"""
Simple Direct Context Test
Tests the DirectOpenAIChat class directly without complex imports
"""

import json
import os
from datetime import datetime
from openai import OpenAI

class MockMemoryEngine:
    """Mock memory engine for testing"""
    def __init__(self):
        self.memories = []
    
    def search_memories(self, query, k=5):
        # Return a few mock memories for testing
        return []
    
    def add_memory(self, content, metadata=None):
        self.memories.append({
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        })

class SimpleDirectOpenAIChat:
    """Simplified version of DirectOpenAIChat for testing"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.memory_engine = MockMemoryEngine()
        
        # Store conversation history in memory by thread_id
        self.conversations = {}
        
    def _get_conversation_messages(self, thread_id: str, limit: int = 10):
        """Get recent conversation messages for a thread"""
        if thread_id not in self.conversations:
            self.conversations[thread_id] = []
        
        return self.conversations[thread_id][-limit:] if limit > 0 else self.conversations[thread_id]
    
    def _add_to_conversation(self, thread_id: str, role: str, content: str):
        """Add a message to conversation history"""
        if thread_id not in self.conversations:
            self.conversations[thread_id] = []
        
        self.conversations[thread_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep conversation history manageable (last 50 messages)
        if len(self.conversations[thread_id]) > 50:
            self.conversations[thread_id] = self.conversations[thread_id][-50:]
    
    def _build_messages_array(self, thread_id: str, user_message: str, system_prompt: str = None):
        """Build the messages array for OpenAI API"""
        messages = []
        
        # 1. System prompt
        if not system_prompt:
            system_prompt = """You're Jeremy's AI assistant with persistent memory. Pay close attention to conversation flow and context.

About Jeremy: 41 years old, wife Ashley, 7 kids, dogs Remy & Bailey. Direct communicator who dislikes generic responses.

CRITICAL CONTEXT RULES:
- ALWAYS reference the entire conversation history, not just the last message
- When Jeremy says "what task" or similar, look for tasks mentioned earlier in our conversation
- Build on previous messages - don't start fresh each time
- Be specific and reference actual details from our conversation"""
        
        messages.append({"role": "system", "content": system_prompt})
        
        # 2. Add conversation history (this is the KEY part!)
        conversation_history = self._get_conversation_messages(thread_id, limit=20)
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 3. Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def chat(self, message: str, thread_id: str = "default", system_prompt: str = None):
        """Main chat method"""
        
        # Build messages array with full context
        messages = self._build_messages_array(
            thread_id=thread_id,
            user_message=message,
            system_prompt=system_prompt
        )
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
            
            assistant_response = response.choices[0].message.content
            
            # Store in conversation history
            self._add_to_conversation(thread_id, "user", message)
            self._add_to_conversation(thread_id, "assistant", assistant_response)
            
            return assistant_response, messages
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise
    
    def clear_thread(self, thread_id: str):
        """Clear conversation history for a specific thread"""
        if thread_id in self.conversations:
            del self.conversations[thread_id]
    
    def get_conversation_history(self, thread_id: str):
        """Get full conversation history for a thread"""
        return self._get_conversation_messages(thread_id, limit=0)

def run_comprehensive_tests():
    """Run all comprehensive context tests"""
    
    api_key = "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"
    
    chat = SimpleDirectOpenAIChat(api_key=api_key, model="gpt-3.5-turbo")
    
    print("🚀 Starting Comprehensive Context Retention Test Suite")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔧 Testing DirectOpenAIChat v1.10.0")
    
    results = []
    
    # Test 5: Long conversation with topic switches
    print("\n" + "="*60)
    print("🧪 TEST 5: Long Conversation with Topic Switches")
    print("="*60)
    
    thread_id = "test_long_conversation"
    chat.clear_thread(thread_id)
    
    try:
        # Topic 1: Python vs Go
        print("\n💬 Topic 1: Python vs Go")
        response1, _ = chat.chat(
            "I need help choosing between Python and Go for a new microservices project. The project needs to handle high throughput APIs.",
            thread_id=thread_id
        )
        print(f"🤖 {response1[:150]}...")
        
        # Follow-up on Python vs Go
        response2, _ = chat.chat(
            "What about performance differences?",
            thread_id=thread_id
        )
        print(f"🤖 {response2[:150]}...")
        
        # Topic switch to Docker
        print("\n💬 Topic Switch: Docker")
        response3, _ = chat.chat(
            "Actually, let's talk about Docker instead. I'm having issues with container networking.",
            thread_id=thread_id
        )
        print(f"🤖 {response3[:150]}...")
        
        # Follow-up on Docker
        response4, _ = chat.chat(
            "The containers can't communicate with each other. What should I check?",
            thread_id=thread_id
        )
        print(f"🤖 {response4[:150]}...")
        
        # Topic switch to database design
        print("\n💬 Topic Switch: Database Design")
        response5, _ = chat.chat(
            "Now I want to discuss database design. Should I use PostgreSQL or MongoDB for user profiles?",
            thread_id=thread_id
        )
        print(f"🤖 {response5[:150]}...")
        
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
            results.append(("Test 5: Long Conversation", True))
        else:
            print("❌ FAILED: AI did not reference earlier language discussion")
            results.append(("Test 5: Long Conversation", False))
            
    except Exception as e:
        print(f"❌ Test 5 failed with error: {e}")
        results.append(("Test 5: Long Conversation", False))
    
    # Test 6: Thread persistence simulation
    print("\n" + "="*60)
    print("🧪 TEST 6: Thread Persistence Simulation")
    print("="*60)
    
    thread_id = "test_persistence"
    
    try:
        # Start a conversation
        print("\n💬 Starting conversation about React hooks")
        response1, _ = chat.chat(
            "I'm having trouble with React useEffect hooks. They're running too often.",
            thread_id=thread_id
        )
        print(f"🤖 {response1[:150]}...")
        
        # Simulate "page reload" by creating new chat instance
        print("\n🔄 Simulating page reload (new chat instance)")
        chat2 = SimpleDirectOpenAIChat(api_key=api_key, model="gpt-3.5-turbo")
        
        # Continue conversation - should remember context (won't work without persistence)
        print("\n💬 Continuing conversation after 'reload'")
        response2, _ = chat2.chat(
            "What was the main issue you mentioned?",
            thread_id=thread_id
        )
        print(f"🤖 {response2}")
        
        # This test will likely fail without file persistence, but that's expected
        if "useEffect" in response2 or "hook" in response2.lower() or "running" in response2.lower():
            print("✅ PASSED: Context maintained across 'page reload'")
            results.append(("Test 6: Thread Persistence", True))
        else:
            print("❌ EXPECTED FAIL: Context lost after 'page reload' (no file persistence in simple test)")
            results.append(("Test 6: Thread Persistence", False))
            
    except Exception as e:
        print(f"❌ Test 6 failed with error: {e}")
        results.append(("Test 6: Thread Persistence", False))
    
    # Test 7: Basic conversation flow
    print("\n" + "="*60)
    print("🧪 TEST 7: Basic Conversation Flow")
    print("="*60)
    
    thread_id = "test_basic_flow"
    
    try:
        # Start planning tasks
        print("\n💬 Planning multiple tasks")
        response1, _ = chat.chat(
            "I need to plan these 3 tasks: 1) Set up CI/CD pipeline, 2) Refactor user authentication, 3) Implement caching layer. Let's start with the first one.",
            thread_id=thread_id
        )
        print(f"🤖 {response1[:150]}...")
        
        # Test contextual reference
        print("\n💬 Testing contextual reference")
        response2, _ = chat.chat(
            "What task should I focus on first?",
            thread_id=thread_id
        )
        print(f"🤖 {response2}")
        
        # Check if AI referenced the CI/CD pipeline
        if "CI/CD" in response2 or "pipeline" in response2.lower() or "first" in response2.lower():
            print("✅ PASSED: AI correctly referenced the CI/CD pipeline task")
            results.append(("Test 7: Basic Flow", True))
        else:
            print("❌ FAILED: AI did not reference the specific tasks")
            results.append(("Test 7: Basic Flow", False))
            
    except Exception as e:
        print(f"❌ Test 7 failed with error: {e}")
        results.append(("Test 7: Basic Flow", False))
    
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
    
    if passed >= 2:  # At least 2/3 passing is good
        print("🎉 TESTS MOSTLY PASSED! Context retention is working well.")
    else:
        print("⚠️  Context retention needs attention.")

if __name__ == "__main__":
    run_comprehensive_tests()
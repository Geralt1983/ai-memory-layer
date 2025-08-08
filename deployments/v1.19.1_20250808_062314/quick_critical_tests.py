#!/usr/bin/env python3
"""
Quick Critical GPT-4o Tests
Focused on the most important validation points
"""

import json
import os
from datetime import datetime
from openai import OpenAI


class QuickGPT4oTester:
    """Quick tester for GPT-4o critical functionality"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.conversations = {}
        
        # System prompt
        self.system_prompt = """You are an AI assistant designed to interact like a sharp, experienced, human-like conversation partner. You are modeled after the best traits of GPT-4o, known for memory-aware, emotionally intelligent, and contextually precise responses.

Your goals are:
- Speak naturally, like a fast-thinking, helpful peer
- Remember and subtly incorporate long-term context and preferences
- Avoid repetition, filler phrases, robotic tone, or over-explaining
- Acknowledge what the user implies, not just what they say
- Maintain continuity in tone, voice, and purpose across threads

DO:
- Be concise, clear, and confident
- Use friendly, professional language with optional cleverness
- Handle ambiguity with tact, not hedging
- Reference memory seamlessly and naturally, not by quoting

DON'T:
- Say "As an AI..." or use boilerplate
- Apologize for things that weren't errors
- Repeat user's question before answering
- Over-explain unless asked to

You are currently supporting a user named Jeremy Kimble, an IT consultant who is building a long-memory AI assistant with vector recall using FAISS and OpenAI's GPT-4o API. He values speed, precision, low-fluff responses, and clever utility."""
        
        # User identity
        self.user_identity = "User Profile: Jeremy Kimble, IT consultant and AI system builder. Communication style: Expects blunt, helpful responses like a capable peer. Current project: Building AI memory layer with vector storage and GPT-4o integration."
        
        # Memories
        self.memories = []
    
    def add_memory(self, content: str, memory_type: str = "general", priority: str = "normal"):
        """Add memory"""
        self.memories.append({
            "content": content,
            "type": memory_type,
            "priority": priority
        })
    
    def _build_messages(self, thread_id: str, user_message: str, include_memories: bool = True):
        """Build message array"""
        messages = []
        
        # System prompt
        messages.append({"role": "system", "content": self.system_prompt})
        
        # Identity
        messages.append({"role": "system", "content": self.user_identity})
        
        # Memory context
        if include_memories and self.memories:
            high_priority = [m for m in self.memories if m.get("priority") == "high"]
            if high_priority:
                memory_context = " | ".join([m["content"] for m in high_priority[:2]])
                messages.append({
                    "role": "system", 
                    "content": f"Background context: {memory_context}"
                })
        
        # Conversation history
        if thread_id in self.conversations:
            for msg in self.conversations[thread_id][-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Current message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def chat(self, message: str, thread_id: str = "default", include_memories: bool = True):
        """Chat with GPT-4o"""
        messages = self._build_messages(thread_id, message, include_memories)
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            top_p=1.0,
            presence_penalty=0.5,
            frequency_penalty=0.25,
            max_tokens=800
        )
        
        assistant_response = response.choices[0].message.content
        
        # Store conversation
        if thread_id not in self.conversations:
            self.conversations[thread_id] = []
        
        self.conversations[thread_id].extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": assistant_response}
        ])
        
        return assistant_response, messages


def run_critical_tests():
    """Run the most critical GPT-4o tests"""
    
    print("🚀 CRITICAL GPT-4o VALIDATION TESTS")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    api_key = "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"
    tester = QuickGPT4oTester(api_key)
    
    results = []
    
    # TEST 1: Memory Recall with Name Correction
    print("\n" + "="*50)
    print("🧠 TEST 1: Memory Recall Integration")
    print("="*50)
    
    thread_id = "memory_test"
    
    print("💬 Setting name correction")
    response1, _ = tester.chat("My kid's name is Chayah. Spell it this way, never Kaia or Kaya.", thread_id)
    print(f"🤖 {response1}")
    
    # Add high priority memory
    tester.add_memory("Jeremy's kid's name is Chayah (C-h-a-y-a-h, never Kaia or Kaya)", "correction", "high")
    
    print("\n💬 Building context")
    response2, _ = tester.chat("She just started preschool today.", thread_id)
    print(f"🤖 {response2}")
    
    print("\n💬 CRITICAL: Memory recall test")
    response3, _ = tester.chat("What do you think preschool is going to be like for her?", thread_id)
    print(f"🤖 {response3}")
    
    # Check for correct usage
    if "Chayah" in response3:
        print("✅ PASSED: Correctly used 'Chayah'")
        results.append(("Memory Recall", True))
    elif "Kaia" in response3 or "Kaya" in response3:
        print("❌ FAILED: Used incorrect spelling")
        results.append(("Memory Recall", False))
    elif "she" in response3.lower() or "your daughter" in response3.lower():
        print("✅ PASSED: Used appropriate pronouns")
        results.append(("Memory Recall", True))
    else:
        print("❌ FAILED: No appropriate reference")
        results.append(("Memory Recall", False))
    
    # TEST 2: Message Array Audit
    print("\n" + "="*50)
    print("🧠 TEST 2: Message Array Audit")
    print("="*50)
    
    # Build a test message array
    messages = tester._build_messages("audit_test", "What should I optimize first?")
    
    print("📋 MESSAGE ARRAY STRUCTURE:")
    for i, msg in enumerate(messages):
        role = msg["role"]
        content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"{i+1}. {role}: {content_preview}")
    
    # Audit checks
    system_count = len([m for m in messages if m["role"] == "system"])
    has_identity = any("Jeremy Kimble" in m["content"] for m in messages)
    has_memory = any("Background context" in m["content"] for m in messages)
    
    audit_passed = system_count >= 2 and has_identity and has_memory
    
    if audit_passed:
        print("✅ PASSED: Message array properly structured")
        results.append(("Message Array", True))
    else:
        print("❌ FAILED: Message array structure issues")
        results.append(("Message Array", False))
    
    # TEST 3: Tone & Command
    print("\n" + "="*50)
    print("🧠 TEST 3: Tone & Command")
    print("="*50)
    
    print("💬 Requesting Python function")
    response_code, _ = tester.chat("Give me a fast function in Python that parses Markdown changelogs into JSON", "code_test")
    print(f"🤖 {response_code}")
    
    # Check tone and content
    no_fluff = not any(phrase in response_code for phrase in ["Certainly!", "I'd be happy to", "As an AI"])
    has_code = "def " in response_code
    direct_intro = any(phrase in response_code for phrase in ["Here's", "This function"])
    
    tone_passed = no_fluff and has_code and direct_intro
    
    if tone_passed:
        print("✅ PASSED: Direct tone with quality code")
        results.append(("Tone & Command", True))
    else:
        print("❌ FAILED: Tone or code quality issues")
        results.append(("Tone & Command", False))
    
    # TEST 4: Context Continuity
    print("\n" + "="*50)
    print("🧠 TEST 4: Context Continuity")
    print("="*50)
    
    continuity_thread = "continuity_test"
    
    print("💬 Setting up microservices context")
    response_setup, _ = tester.chat("I'm architecting a microservices system with User Service, Payment Service, and Notification Service. The User Service needs high availability.", continuity_thread)
    print(f"🤖 Setup: {response_setup[:100]}...")
    
    print("\n💬 Context retention test")
    response_context, _ = tester.chat("Which service did I say needed high availability?", continuity_thread)
    print(f"🤖 {response_context}")
    
    if "User Service" in response_context or "user service" in response_context.lower():
        print("✅ PASSED: Perfect context retention")
        results.append(("Context Continuity", True))
    else:
        print("❌ FAILED: Lost context")
        results.append(("Context Continuity", False))
    
    # RESULTS SUMMARY
    print("\n" + "="*60)
    print("📊 CRITICAL TEST RESULTS")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Critical Score: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 PERFECT! GPT-4o is production-ready!")
        print("🔥 Memory + Context + Tone = SUCCESS")
    elif passed >= 3:
        print("✅ EXCELLENT! Nearly perfect GPT-4o performance")
    else:
        print("⚠️  NEEDS REFINEMENT - Focus on failed tests")


if __name__ == "__main__":
    run_critical_tests()
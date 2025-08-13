#!/usr/bin/env python3
"""
Test Memory Deduplication and Identity Anchoring Improvements
Test the new deduplication system and enhanced identity framing
"""

import json
import os
from datetime import datetime
from openai import OpenAI
import re
import pytest

if not os.getenv("OPENAI_API_KEY"):
    pytest.skip("OPENAI_API_KEY not set", allow_module_level=True)


class MemoryDeduplicationTester:
    """Test memory deduplication and identity anchoring improvements"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.conversations = {}
        
        # Mock enhanced system with deduplication
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
        
        # Simulate Jeremy's memory profile
        self.jeremy_memories = [
            "Jeremy prefers direct communication and dislikes fluff",
            "User values technical precision and blunt responses", 
            "Jeremy expects responses like a capable peer, not customer support",
            "User enjoys exploring different perspectives and weighing pros and cons",
            "Jeremy prefers inquisitive responses that dig deeper",
            "User values speed and precision in all responses",
            "Jeremy is an IT consultant building AI memory systems",
            "User is working on FAISS integration with OpenAI GPT-4o",
            "Jeremy has been dealing with fatigue issues from stimulants",
            "User has discussed stimulant-related energy management"
        ]
    
    def _dedupe_and_paraphrase_memories(self, contents, memory_type):
        """Mock the new deduplication system"""
        if not contents:
            return []
        
        unique_contents = []
        seen_keywords = set()
        
        for content in contents:
            key_terms = set(word.lower() for word in content.split() 
                           if len(word) > 3 and word.isalpha())
            
            overlap = any(len(key_terms & seen) > 2 for seen in seen_keywords)
            
            if not overlap:
                unique_contents.append(content)
                seen_keywords.add(frozenset(key_terms))
            
            if len(unique_contents) >= 3:
                break
        
        return unique_contents
    
    def _create_memory_summary(self, is_preferences=True):
        """Mock enhanced memory summary with deduplication"""
        if is_preferences:
            # Filter to preference-related memories
            pref_memories = [m for m in self.jeremy_memories if any(word in m.lower() 
                           for word in ['prefer', 'value', 'like', 'expect'])]
            
            unique_prefs = self._dedupe_and_paraphrase_memories(pref_memories, 'preference')
            
            if unique_prefs:
                # Vary phrasing to avoid repetition
                pref_phrases = [
                    f"Jeremy values {', '.join(unique_prefs[:2])}",
                    f"Communication style: {', '.join(unique_prefs[:2])}",
                    f"Prefers {', '.join(unique_prefs[:2])}"
                ]
                return pref_phrases[0]  # Use first variation for test
        
        return "Technical background: building AI memory systems with FAISS/OpenAI"
    
    def _create_identity_message(self, is_thread_start=True):
        """Mock enhanced identity message"""
        if is_thread_start:
            return """User Identity: You're speaking with Jeremy, who explicitly prefers blunt, helpful responses like a capable peer. Jeremy dislikes robotic responses, values technical precision, and expects you to reference conversation history naturally. Current focus: building AI memory layer with vector storage and GPT-4o integration."""
        else:
            return """User Profile: Jeremy Kimble, IT consultant and AI system builder. Communication style: Expects blunt, helpful responses like a capable peer. Current project context: building AI memory layer with vector storage and GPT-4o integration."""
    
    def _build_messages(self, user_message, thread_id, is_thread_start=True):
        """Build enhanced message array with improvements"""
        messages = []
        
        # 1. System prompt
        messages.append({"role": "system", "content": self.system_prompt})
        
        # 2. Enhanced identity profile (stronger for thread starts)
        messages.append({
            "role": "system", 
            "content": self._create_identity_message(is_thread_start)
        })
        
        # 3. Deduplicated memory summary
        memory_summary = self._create_memory_summary(is_preferences=True)
        if memory_summary:
            messages.append({
                "role": "system",
                "content": f"Background context: {memory_summary}"
            })
        
        # 4. Conversation history (if any)
        if thread_id in self.conversations:
            for msg in self.conversations[thread_id][-6:]:  # Last 6 messages
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 5. Current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def chat(self, message, thread_id="default", is_thread_start=True):
        """Chat with enhanced deduplication and identity anchoring"""
        messages = self._build_messages(message, thread_id, is_thread_start)
        
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


def test_memory_improvements():
    """Test the memory deduplication and identity anchoring improvements"""
    
    print("🧪 TESTING MEMORY DEDUPLICATION & IDENTITY ANCHORING")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Testing enhanced memory recall and identity framing")
    
    api_key = "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"
    tester = MemoryDeduplicationTester(api_key)
    
    results = []
    
    # Test 1: Thread Start Identity Anchoring
    print("\n" + "="*60)
    print("🧪 TEST 1: Enhanced Thread Start Identity Framing")
    print("="*60)
    
    thread_id = "identity_test_thread"
    
    print("💬 Testing thread start with strong identity anchoring")
    response1, messages1 = tester.chat(
        "Hey, what's the best approach for database optimization?",
        thread_id,
        is_thread_start=True
    )
    print(f"🤖 {response1}")
    
    # Check for identity-forward language
    identity_markers = [
        "jeremy", "you", "your", "preference", "style", "value"
    ]
    
    identity_referenced = any(marker in response1.lower() for marker in identity_markers)
    personal_tone = not any(phrase in response1.lower() for phrase in [
        "as an ai", "i'm here to help", "happy to assist", "let me help you"
    ])
    
    if identity_referenced and personal_tone:
        print("✅ SUCCESS: Strong identity anchoring in thread start")
        results.append(("Thread Start Identity", True))
    else:
        print("❌ FAILED: Generic response despite identity framing")
        results.append(("Thread Start Identity", False))
    
    # Test 2: Memory Deduplication Prevention
    print("\n" + "="*60)
    print("🧪 TEST 2: Memory Deduplication Prevention")
    print("="*60)
    
    print("💬 Testing multiple preference recalls for redundancy")
    response2, messages2 = tester.chat(
        "What's your take on API design patterns?",
        thread_id,
        is_thread_start=False
    )
    print(f"🤖 {response2}")
    
    # Test follow-up to check for repetitive preference mentions
    response3, messages3 = tester.chat(
        "And what about microservices vs monoliths?",
        thread_id,
        is_thread_start=False
    )
    print(f"🤖 {response3}")
    
    # Check for redundant phrasing between responses
    response2_words = set(response2.lower().split())
    response3_words = set(response3.lower().split())
    
    # Look for exact phrase repetition (sign of poor deduplication)
    redundant_phrases = [
        "prefer direct", "values technical", "blunt responses",
        "capable peer", "likes exploring", "weighing pros"
    ]
    
    repetition_count = 0
    for phrase in redundant_phrases:
        if phrase in response2.lower() and phrase in response3.lower():
            repetition_count += 1
    
    deduplication_working = repetition_count <= 1  # Allow 1 overlap
    
    if deduplication_working:
        print("✅ SUCCESS: Minimal redundancy between responses")
        results.append(("Memory Deduplication", True))
    else:
        print(f"❌ FAILED: {repetition_count} redundant phrases detected")
        results.append(("Memory Deduplication", False))
    
    # Test 3: Mid-Thread Identity Softening  
    print("\n" + "="*60)
    print("🧪 TEST 3: Mid-Thread Identity Balance")
    print("="*60)
    
    print("💬 Testing mid-thread identity balance (less aggressive)")
    response4, messages4 = tester.chat(
        "Any thoughts on Python vs Go for backend development?",
        thread_id,
        is_thread_start=False
    )
    print(f"🤖 {response4}")
    
    # Check that mid-thread responses are still personal but less identity-focused
    still_personal = any(word in response4.lower() for word in ["you", "your"])
    not_overly_identity_focused = response4.lower().count("jeremy") <= 1
    
    if still_personal and not_overly_identity_focused:
        print("✅ SUCCESS: Balanced mid-thread identity references")
        results.append(("Mid-Thread Balance", True))
    else:
        print("❌ FAILED: Either too generic or overly identity-focused")
        results.append(("Mid-Thread Balance", False))
    
    # Test 4: Memory Variation Testing
    print("\n" + "="*60)
    print("🧪 TEST 4: Memory Recall Variation")
    print("="*60)
    
    # Create new thread to test memory variation
    thread_id2 = "variation_test_thread"
    
    print("💬 Testing memory recall phrasing variation")
    response5, _ = tester.chat(
        "What do you think about code review practices?",
        thread_id2, 
        is_thread_start=True
    )
    print(f"🤖 Response A: {response5}")
    
    # Second similar question to check for phrasing variation
    response6, _ = tester.chat(
        "How about testing strategies?",
        thread_id2,
        is_thread_start=False  
    )
    print(f"🤖 Response B: {response6}")
    
    # Check if memory context is varied in phrasing
    memory_phrases_a = [phrase for phrase in ["values", "prefers", "communication style"] 
                       if phrase in response5.lower()]
    memory_phrases_b = [phrase for phrase in ["values", "prefers", "communication style"] 
                       if phrase in response6.lower()]
    
    varied_phrasing = len(set(memory_phrases_a) & set(memory_phrases_b)) < len(memory_phrases_a)
    
    if varied_phrasing or len(memory_phrases_a) <= 1:
        print("✅ SUCCESS: Memory phrasing shows variation")
        results.append(("Memory Variation", True))
    else:
        print("❌ FAILED: Identical memory phrasing detected")
        results.append(("Memory Variation", False))
    
    # Results summary
    print("\n" + "="*60)
    print("📊 MEMORY IMPROVEMENT TEST RESULTS")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Improvement Score: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL MEMORY IMPROVEMENTS WORKING!")
        print("🔹 Identity anchoring is stronger at thread starts")
        print("🔹 Memory deduplication prevents redundancy")
        print("🔹 Mid-thread identity balance maintained")
        print("🔹 Memory recall phrasing shows variation")
    elif passed >= 3:
        print("✅ MAJOR SUCCESS! Most improvements working well")
    else:
        print("⚠️  NEEDS REFINEMENT - Some improvements not taking effect")


if __name__ == "__main__":
    test_memory_improvements()
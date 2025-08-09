#!/usr/bin/env python3
"""
Simple GPT-4o Enhancement Test
Direct test of GPT-4o capabilities without complex imports
"""

import json
import os
from datetime import datetime
from openai import OpenAI


class SimpleGPT4oChat:
    """Simplified GPT-4o chat with enhancements for testing"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.conversations = {}
        
        # Load optimized system prompt
        self.system_prompt = self._load_system_prompt()
        
        # User identity for personalization
        self.user_identity = "User Profile: Jeremy Kimble, IT consultant and AI system builder. Communication style: Expects blunt, helpful responses like a capable peer. Current project: Building AI memory layer with vector storage and GPT-4o integration."
        
        # Behavior expectations
        self.behavior_log = "Past interaction patterns: Avoid caveats and hedge words unless uncertainty is genuine | Provide formatted code when asked, with brief explanations | Speak like a capable peer, not customer support | Reference conversation history naturally without explicit callbacks"
    
    def _load_system_prompt(self):
        """Load GPT-4o optimized system prompt"""
        prompt_path = "./prompts/system_prompt_4o.txt"
        if os.path.exists(prompt_path):
            try:
                with open(prompt_path, 'r') as f:
                    return f.read().strip()
            except:
                pass
        
        # Fallback prompt
        return """You are an AI assistant designed to interact like a sharp, experienced, human-like conversation partner. You are modeled after the best traits of GPT-4o, known for memory-aware, emotionally intelligent, and contextually precise responses.

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
    
    def _build_messages(self, thread_id: str, user_message: str):
        """Build GPT-4o optimized message array"""
        messages = []
        
        # 1. Primary system prompt
        messages.append({"role": "system", "content": self.system_prompt})
        
        # 2. Identity profile
        messages.append({"role": "system", "content": self.user_identity})
        
        # 3. Behavior log
        messages.append({"role": "system", "content": self.behavior_log})
        
        # 4. Conversation history
        if thread_id in self.conversations:
            for msg in self.conversations[thread_id][-10:]:  # Last 10 messages
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 5. Current message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def chat(self, message: str, thread_id: str = "default"):
        """GPT-4o optimized chat method"""
        messages = self._build_messages(thread_id, message)
        
        try:
            # GPT-4o optimized API call
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                top_p=1.0,
                presence_penalty=0.5,
                frequency_penalty=0.25,
                max_tokens=1200
            )
            
            assistant_response = response.choices[0].message.content
            
            # Store conversation
            if thread_id not in self.conversations:
                self.conversations[thread_id] = []
            
            self.conversations[thread_id].extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": assistant_response}
            ])
            
            return assistant_response
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise
    
    def clear_thread(self, thread_id: str):
        """Clear conversation thread"""
        if thread_id in self.conversations:
            del self.conversations[thread_id]


def test_gpt4o_enhancements():
    """Test GPT-4o enhancements"""
    
    print("🚀 Testing GPT-4o Enhancements")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    api_key = "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"
    chat = SimpleGPT4oChat(api_key)
    
    results = []
    
    # Test 1: Direct, confident responses
    print("\n" + "="*50)
    print("🧠 TEST 1: Human-like Response Quality")
    print("="*50)
    
    response1 = chat.chat(
        "Should I use React or Vue for my next project?",
        "test_confidence"
    )
    print(f"🤖 {response1}")
    
    # Check for confidence (no excessive hedging)
    hedging_phrases = ["it depends", "you might want to consider", "perhaps", "possibly", "it could be that"]
    has_excessive_hedging = sum(1 for phrase in hedging_phrases if phrase in response1.lower()) > 1
    
    if not has_excessive_hedging and len(response1.split()) < 200:
        print("✅ PASSED: Response is direct and confident")
        results.append(("Confident Responses", True))
    else:
        print("❌ FAILED: Response has excessive hedging or verbosity")
        results.append(("Confident Responses", False))
    
    # Test 2: Context continuity
    print("\n" + "="*50)
    print("🧠 TEST 2: Context Continuity")
    print("="*50)
    
    # Set up context
    response2a = chat.chat(
        "I'm building a microservices architecture with these services: User Service, Payment Service, and Notification Service. The User Service needs high availability.",
        "test_context"
    )
    print(f"🤖 Setup: {response2a[:100]}...")
    
    # Test context retention
    response2b = chat.chat(
        "Which service did I say needed high availability?",
        "test_context"
    )
    print(f"🤖 Context test: {response2b}")
    
    if "User Service" in response2b or "user service" in response2b.lower():
        print("✅ PASSED: Context correctly maintained")
        results.append(("Context Continuity", True))
    else:
        print("❌ FAILED: Context not maintained")
        results.append(("Context Continuity", False))
    
    # Test 3: No AI boilerplate
    print("\n" + "="*50)
    print("🧠 TEST 3: Natural Communication Style")
    print("="*50)
    
    response3 = chat.chat(
        "I'm feeling overwhelmed with all these technical decisions.",
        "test_style"
    )
    print(f"🤖 {response3}")
    
    # Check for AI boilerplate
    ai_phrases = ["as an ai", "i'm here to help", "i understand you're feeling", "as a language model"]
    has_ai_boilerplate = any(phrase in response3.lower() for phrase in ai_phrases)
    
    if not has_ai_boilerplate:
        print("✅ PASSED: Natural, peer-like communication")
        results.append(("Natural Style", True))
    else:
        print("❌ FAILED: Contains AI boilerplate")
        results.append(("Natural Style", False))
    
    # Test 4: Technical competence
    print("\n" + "="*50)
    print("🧠 TEST 4: Technical Competence")
    print("="*50)
    
    response4 = chat.chat(
        "What's the best approach for handling database transactions in a distributed system?",
        "test_technical"
    )
    print(f"🤖 {response4}")
    
    # Check for technical depth
    technical_terms = ["saga", "two-phase commit", "eventual consistency", "distributed", "transaction", "acid"]
    has_technical_depth = sum(1 for term in technical_terms if term in response4.lower()) >= 2
    
    if has_technical_depth and len(response4.split()) > 50:
        print("✅ PASSED: Demonstrates technical competence")
        results.append(("Technical Competence", True))
    else:
        print("❌ FAILED: Lacks technical depth")
        results.append(("Technical Competence", False))
    
    # Results summary
    print("\n" + "="*50)
    print("📊 GPT-4o ENHANCEMENT RESULTS")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed >= 3:
        print("🎉 GPT-4o ENHANCEMENTS WORKING EXCELLENTLY!")
        print("🔥 Ready for human-like, sophisticated AI interactions!")
    elif passed >= 2:
        print("✅ GPT-4o enhancements mostly working")
    else:
        print("⚠️  GPT-4o enhancements need refinement")


if __name__ == "__main__":
    test_gpt4o_enhancements()
#!/usr/bin/env python3
"""
Comprehensive GPT-4o Test Suite
Implements the 6 critical tests for production-ready GPT-4o behavior
"""

import json
import os
from datetime import datetime
from openai import OpenAI


class ComprehensiveGPT4oTester:
    """Comprehensive tester for GPT-4o implementation"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.conversations = {}
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
        
        # User identity for personalization
        self.user_identity = "User Profile: Jeremy Kimble, IT consultant and AI system builder. Communication style: Expects blunt, helpful responses like a capable peer. Current project: Building AI memory layer with vector storage and GPT-4o integration."
        
        # Behavior expectations
        self.behavior_log = "Past interaction patterns: Avoid caveats and hedge words unless uncertainty is genuine | Provide formatted code when asked, with brief explanations | Speak like a capable peer, not customer support | Reference conversation history naturally without explicit callbacks"
        
        # Mock memory system for testing
        self.memories = []
    
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
- Acknowledge what the user implies, not just what you say
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
    
    def add_memory(self, content: str, memory_type: str = "general", priority: str = "normal"):
        """Add a memory to the mock memory system"""
        self.memories.append({
            "content": content,
            "type": memory_type,
            "priority": priority,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_relevant_memories(self, query: str, limit: int = 3):
        """Get relevant memories (simple keyword matching for testing)"""
        relevant = []
        query_lower = query.lower()
        
        for memory in self.memories:
            if any(word in memory["content"].lower() for word in query_lower.split()):
                relevant.append(memory)
                if len(relevant) >= limit:
                    break
        
        return relevant
    
    def _build_messages(self, thread_id: str, user_message: str, include_memories: bool = True):
        """Build GPT-4o optimized message array"""
        messages = []
        
        # 1. Primary system prompt
        messages.append({"role": "system", "content": self.system_prompt})
        
        # 2. Identity profile
        messages.append({"role": "system", "content": self.user_identity})
        
        # 3. Behavior log
        messages.append({"role": "system", "content": self.behavior_log})
        
        # 4. Memory context (if available and requested)
        if include_memories and self.memories:
            relevant_memories = self.get_relevant_memories(user_message)
            if relevant_memories:
                memory_context = " | ".join([
                    f"{mem['type']}: {mem['content']}" for mem in relevant_memories[:3]
                ])
                messages.append({
                    "role": "system", 
                    "content": f"Background context: {memory_context}"
                })
        
        # 5. Conversation history
        if thread_id in self.conversations:
            for msg in self.conversations[thread_id][-10:]:  # Last 10 messages
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 6. Current message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def chat(self, message: str, thread_id: str = "default", include_memories: bool = True):
        """GPT-4o chat with comprehensive context"""
        messages = self._build_messages(thread_id, message, include_memories)
        
        try:
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
            
            return assistant_response, messages
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise
    
    def clear_thread(self, thread_id: str):
        """Clear conversation thread"""
        if thread_id in self.conversations:
            del self.conversations[thread_id]


def test_1_memory_recall_integration():
    """TEST 1: Memory Recall Integration with Specific Name Correction"""
    print("\n" + "="*60)
    print("🧠 TEST 1: Memory Recall Integration")
    print("="*60)
    
    api_key = "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"
    tester = ComprehensiveGPT4oTester(api_key)
    
    thread_id = "memory_recall_test"
    
    # Message 1: Name correction
    print("💬 Message 1: Name correction")
    response1, _ = tester.chat(
        "My kid's name is Chayah. Spell it this way, never Kaia or Kaya.",
        thread_id
    )
    print(f"🤖 {response1}")
    
    # Add this as a high-priority memory
    tester.add_memory(
        "Jeremy's kid's name is Chayah (spelled C-h-a-y-a-h, never Kaia or Kaya)", 
        "correction", 
        "high"
    )
    
    # Message 2: Context building
    print("\n💬 Message 2: Context building")
    response2, _ = tester.chat(
        "She just started preschool today.",
        thread_id
    )
    print(f"🤖 {response2}")
    
    # Message 3: Memory recall test
    print("\n💬 Message 3: Memory recall test (CRITICAL)")
    response3, _ = tester.chat(
        "What do you think preschool is going to be like for her?",
        thread_id
    )
    print(f"🤖 {response3}")
    
    # Check for correct name usage
    if "Chayah" in response3:
        print("✅ PASSED: Correctly used 'Chayah' name")
        return True
    elif "Kaia" in response3 or "Kaya" in response3:
        print("❌ FAILED: Used incorrect name spelling (Kaia/Kaya)")
        return False
    elif "your daughter" in response3.lower() or "she" in response3.lower():
        print("✅ PASSED: Used pronouns appropriately without name confusion")
        return True
    else:
        print("❌ FAILED: Did not reference the child appropriately")
        return False


def test_2_token_budget_message_window():
    """TEST 2: Token Budget + Message Window Check"""
    print("\n" + "="*60)
    print("🧠 TEST 2: Token Budget + Message Window Check")
    print("="*60)
    
    api_key = "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"
    tester = ComprehensiveGPT4oTester(api_key)
    
    thread_id = "token_budget_test"
    
    # Send 30+ messages to test context window management
    topics = [
        "React vs Vue framework comparison",
        "Docker container networking issues", 
        "PostgreSQL vs MongoDB for user data",
        "Microservices architecture patterns",
        "CI/CD pipeline optimization",
        "Kubernetes deployment strategies",
        "API rate limiting implementations",
        "Database indexing performance",
        "Redis caching strategies",
        "Load balancer configuration",
        "SSL certificate management",
        "OAuth2 implementation details",
        "GraphQL vs REST API design",
        "WebSocket connection scaling",
        "CDN optimization techniques"
    ]
    
    # Rapid-fire conversation
    print("💬 Sending 30+ messages rapidly...")
    for i, topic in enumerate(topics):
        if i < 10:  # Show first 10 for brevity
            print(f"Message {i+1}: {topic}")
        response, _ = tester.chat(f"Quick thoughts on {topic}?", thread_id)
        if i < 3:  # Show first few responses
            print(f"🤖 {response[:100]}...")
        
        # Add follow-up questions
        tester.chat(f"What's the main benefit of that approach?", thread_id)
        
        if i >= 14:  # Stop after 30 total messages
            break
    
    print(f"📊 Sent 30+ messages. Total conversation length: {len(tester.conversations[thread_id])}")
    
    # Final test: Can it summarize coherently?
    print("\n💬 Final coherence test")
    response_final, _ = tester.chat("What have we talked about?", thread_id)
    print(f"🤖 {response_final}")
    
    # Check response quality
    if (len(response_final.split()) > 30 and 
        len(response_final.split()) < 200 and
        any(topic.split()[0].lower() in response_final.lower() for topic in topics[:5])):
        print("✅ PASSED: Coherent summary within reasonable token budget")
        return True
    else:
        print("❌ FAILED: Summary too short, too long, or missing key topics")
        return False


def test_3_message_array_audit():
    """TEST 3: Message Array Audit"""
    print("\n" + "="*60)
    print("🧠 TEST 3: Message Array Audit")
    print("="*60)
    
    api_key = "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"
    tester = ComprehensiveGPT4oTester(api_key)
    
    thread_id = "message_audit_test"
    
    # Add some memories
    tester.add_memory("Jeremy prefers TypeScript for type safety", "preference", "high")
    tester.add_memory("Current project uses FAISS vector storage", "technical", "normal")
    
    # Build a conversation
    tester.chat("I'm working on database optimization", thread_id)
    tester.chat("The queries are running slowly", thread_id)
    
    # Get the message array for audit
    print("💬 Building message array for audit")
    test_message = "What indexing strategy should I use?"
    messages = tester._build_messages(thread_id, test_message)
    
    # Dump the message array
    print("\n📋 MESSAGE ARRAY AUDIT:")
    print(json.dumps(messages, indent=2))
    
    # Analyze the structure
    system_messages = [msg for msg in messages if msg["role"] == "system"]
    user_messages = [msg for msg in messages if msg["role"] == "user"]
    assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
    
    print(f"\n📊 STRUCTURE ANALYSIS:")
    print(f"- Total messages: {len(messages)}")
    print(f"- System messages: {len(system_messages)}")
    print(f"- User messages: {len(user_messages)}")
    print(f"- Assistant messages: {len(assistant_messages)}")
    
    # Check requirements
    checks = {
        "Has system prompt": len(system_messages) >= 1 and "assistant designed to interact" in system_messages[0]["content"],
        "Has identity profile": any("Jeremy Kimble" in msg["content"] for msg in system_messages),
        "Has behavior log": any("interaction patterns" in msg["content"] for msg in system_messages),
        "Has memory context": any("Background context" in msg["content"] for msg in system_messages),
        "Recent conversation history": len(user_messages) >= 2,
        "No raw vector dumps": not any("Vector(id=" in msg["content"] for msg in messages),
        "Natural language memories": any("TypeScript" in msg["content"] for msg in system_messages if "Background context" in msg["content"])
    }
    
    print(f"\n✅ AUDIT RESULTS:")
    all_passed = True
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"  
        print(f"{status}: {check}")
        if not passed:
            all_passed = False
    
    return all_passed


def test_4_tone_command():
    """TEST 4: Tone + Command Test"""
    print("\n" + "="*60)
    print("🧠 TEST 4: Tone + Command Test")
    print("="*60)
    
    api_key = "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"
    tester = ComprehensiveGPT4oTester(api_key)
    
    thread_id = "tone_command_test"
    
    print("💬 Requesting Python function")
    response, _ = tester.chat(
        "Give me a fast function in Python that parses Markdown changelogs into JSON",
        thread_id
    )
    print(f"🤖 {response}")
    
    # Check tone and content
    checks = {
        "No 'Certainly!' or AI fluff": not any(phrase in response for phrase in ["Certainly!", "I'd be happy to", "As an AI"]),
        "Direct intro": any(phrase in response for phrase in ["Here's a", "Here's the", "This function", "def "]),
        "Contains actual code": "def " in response and "import" in response,
        "Has clear explanation": len(response.split()) > 50,
        "Concise (not verbose)": len(response.split()) < 400
    }
    
    print(f"\n✅ TONE & COMMAND ANALYSIS:")
    all_passed = True
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check}")
        if not passed:
            all_passed = False
    
    return all_passed


def test_5_fallback():
    """TEST 5: Fallback Test (Memory Failure)"""
    print("\n" + "="*60)
    print("🧠 TEST 5: Fallback Test")
    print("="*60)
    
    api_key = "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"
    tester = ComprehensiveGPT4oTester(api_key)
    
    thread_id = "fallback_test"
    
    # Test with no memories and no conversation history
    print("💬 Testing with no memory context")
    response, _ = tester.chat(
        "What's the best approach for handling distributed transactions?",
        thread_id,
        include_memories=False  # Disable memory injection
    )
    print(f"🤖 {response}")
    
    # Check that it still works fine
    checks = {
        "No errors or empty response": len(response.strip()) > 50,
        "Technical competence": any(term in response.lower() for term in ["saga", "two-phase", "acid", "consistency", "distributed"]),
        "No hallucination about memory": "previously discussed" not in response.lower() and "as we talked about" not in response.lower(),
        "Natural tone maintained": not any(phrase in response for phrase in ["I don't have context", "No information available"])
    }
    
    print(f"\n✅ FALLBACK TEST ANALYSIS:")
    all_passed = True
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check}")
        if not passed:
            all_passed = False
    
    return all_passed


def test_6_thread_persistence():
    """TEST 6: Thread_ID Persistence"""
    print("\n" + "="*60)
    print("🧠 TEST 6: Thread_ID Persistence")
    print("="*60)
    
    api_key = "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"
    tester = ComprehensiveGPT4oTester(api_key)
    
    # Create an old thread with specific content
    old_thread_id = "old_project_thread"
    
    print("💬 Creating old thread context")
    tester.chat("I'm working on implementing OAuth2 authentication", old_thread_id)
    tester.chat("Using JWT tokens for session management", old_thread_id)
    tester.chat("Having issues with token refresh logic", old_thread_id)
    
    print("📊 Old thread established with OAuth2 context")
    
    # Simulate "reopening" the thread after time has passed
    print("\n💬 Reopening old thread")
    response, _ = tester.chat(
        "Remind me what we were working on here?",
        old_thread_id
    )
    print(f"🤖 {response}")
    
    # Check for context recall
    oauth_terms = ["oauth", "jwt", "token", "authentication", "auth"]
    mentions_context = any(term in response.lower() for term in oauth_terms)
    
    if mentions_context:
        print("✅ PASSED: Successfully recalled thread context")
        return True
    else:
        print("❌ FAILED: Did not recall thread-specific context")
        return False


def main():
    """Run comprehensive GPT-4o test suite"""
    print("🚀 COMPREHENSIVE GPT-4o TEST SUITE")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔥 Testing Production-Ready GPT-4o Behavior")
    print("🎯 Claude = Code Monkey | GPT-4o = Production Brains")
    
    # Ensure directories exist
    os.makedirs("./prompts", exist_ok=True)
    
    results = []
    
    try:
        print("\n🏁 Running 6 Critical Tests...")
        results.append(("Memory Recall Integration", test_1_memory_recall_integration()))
        results.append(("Token Budget + Message Window", test_2_token_budget_message_window()))
        results.append(("Message Array Audit", test_3_message_array_audit()))
        results.append(("Tone + Command", test_4_tone_command()))
        results.append(("Fallback Test", test_5_fallback()))
        results.append(("Thread_ID Persistence", test_6_thread_persistence()))
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        return
    
    # Results summary
    print("\n" + "="*60)
    print("📊 FINAL GPT-4o PRODUCTION READINESS REPORT")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Score: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 PERFECT SCORE! GPT-4o is production-ready!")
        print("🔥 Human-like responses + memory integration + context retention = SUCCESS")
        print("🚀 Ready to deploy and deliver superior AI experience!")
    elif passed >= 5:
        print("✅ EXCELLENT! GPT-4o is nearly production-ready")
        print("🔧 Minor tweaks needed for perfect performance")
    elif passed >= 3:
        print("⚠️  GOOD progress, but needs refinement")
        print("🛠️  Focus on failed tests for production readiness")
    else:
        print("❌ NEEDS WORK - Core functionality issues detected")
        print("🔧 Review system prompts, memory integration, and context building")


if __name__ == "__main__":
    main()
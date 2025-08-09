#!/usr/bin/env python3
"""
Test Semantic Drift Fixes
Test the context anchoring system for ambiguous follow-ups like "what do you think"
"""

import json
import os
from datetime import datetime
from openai import OpenAI
import re


class SemanticDriftTester:
    """Test semantic drift prevention with context anchoring"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.conversations = {}
        
        # Mock the semantic drift fix methods
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
        
        self.user_identity = "User Profile: Jeremy Kimble, IT consultant and AI system builder. Communication style: Expects blunt, helpful responses like a capable peer. Current project: Building AI memory layer with vector storage and GPT-4o integration."
        
        # Ambiguous follow-up patterns
        self.ambiguous_patterns = [
            r"^what (about|do you think|should I do)",
            r"^yeah(?!\\w)",
            r"^sure(?!\\w)",
            r"^ok(?!\\w)",
            r"^maybe(?!\\w)",
            r"^probably(?!\\w)",
            r"^that one",
            r"^not really",
            r"^I guess",
            r"^makes sense",
            r"^kind of",
            r"^a bit",
            r"^me too",
            r"^same here",
            r"^you actually",
            r"^I agree",
            r"^that too",
            r"^it does",
            r"^it doesn't",
            r"^neither",
            r"^both",
            r"^either",
            r"^depends",
            r"^interesting",
            r"^cool",
            r"^true",
            r"^false",
            r"^no(?!\\w)",
            r"^yes(?!\\w)",
            r"^I don't know",
            r"^not sure",
            r"^you're right",
            r"^maybe not",
        ]
    
    def _get_conversation_messages(self, thread_id: str, limit: int = 10):
        """Get recent conversation messages"""
        if thread_id not in self.conversations:
            self.conversations[thread_id] = []
        
        effective_limit = max(limit, 6) if limit > 0 else 0
        return self.conversations[thread_id][-effective_limit:] if effective_limit > 0 else self.conversations[thread_id]
    
    def _is_ambiguous_followup(self, user_input: str) -> bool:
        """Detect if user input is an ambiguous follow-up that needs context anchoring"""
        user_lower = user_input.lower().strip()
        
        for pattern in self.ambiguous_patterns:
            if re.match(pattern, user_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _get_last_assistant_message(self, thread_id: str):
        """Get the most recent assistant message from conversation history"""
        recent_messages = self._get_conversation_messages(thread_id, limit=10)
        
        for msg in reversed(recent_messages):
            if msg["role"] == "assistant":
                return msg["content"]
        
        return None
    
    def _create_context_anchor(self, user_input: str, thread_id: str):
        """Create a context anchoring system message for ambiguous follow-ups"""
        if not self._is_ambiguous_followup(user_input):
            return None
        
        last_ai_message = self._get_last_assistant_message(thread_id)
        if not last_ai_message:
            return None
        
        # For very vague responses, add more context
        extra_vague_patterns = [r"^what do you think", r"^what about", r"^sure$", r"^yeah$", r"^depends"]
        is_extra_vague = any(re.match(pattern, user_input.lower().strip(), re.IGNORECASE) for pattern in extra_vague_patterns)
        
        if is_extra_vague:
            # Get the last user-assistant exchange for richer context
            recent_messages = self._get_conversation_messages(thread_id, limit=6)
            if len(recent_messages) >= 2:
                # Find the last user message and assistant response pair
                last_user_msg = None
                for msg in reversed(recent_messages):
                    if msg["role"] == "user":
                        last_user_msg = msg["content"]
                        break
                
                if last_user_msg:
                    anchor_msg = f"CONTEXT ANCHOR: User's vague reply '{user_input}' refers to this conversation thread: User said '{last_user_msg[:150]}{'...' if len(last_user_msg) > 150 else ''}' and Assistant replied '{last_ai_message[:150]}{'...' if len(last_ai_message) > 150 else ''}'. Stay focused on this specific topic."
                else:
                    anchor_msg = f"CONTEXT ANCHOR: User's reply '{user_input}' is in response to the previous assistant message: '{last_ai_message[:200]}{'...' if len(last_ai_message) > 200 else ''}'"
            else:
                anchor_msg = f"CONTEXT ANCHOR: User's reply '{user_input}' is in response to the previous assistant message: '{last_ai_message[:200]}{'...' if len(last_ai_message) > 200 else ''}'"
        else:
            # Standard anchoring for less vague responses
            anchor_msg = f"CONTEXT ANCHOR: User's reply '{user_input}' is in response to the previous assistant message: '{last_ai_message[:200]}{'...' if len(last_ai_message) > 200 else ''}'"
        
        return anchor_msg
    
    def _build_messages(self, thread_id: str, user_message: str):
        """Build GPT-4o message array with semantic drift fixes"""
        messages = []
        
        # 1. System prompt
        messages.append({"role": "system", "content": self.system_prompt})
        
        # 2. Identity profile
        messages.append({"role": "system", "content": self.user_identity})
        
        # 3. Context anchoring for ambiguous follow-ups (CRITICAL for semantic drift prevention)
        context_anchor = self._create_context_anchor(user_message, thread_id)
        if context_anchor:
            messages.append({
                "role": "system",
                "content": context_anchor
            })
            print(f"🔗 ANCHOR APPLIED: {context_anchor}")
        
        # 4. Conversation history
        conversation_history = self._get_conversation_messages(thread_id, limit=20)
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 5. Current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def chat(self, message: str, thread_id: str = "default"):
        """Chat with semantic drift prevention"""
        messages = self._build_messages(thread_id, message)
        
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


def test_semantic_drift_fixes():
    """Test the specific semantic drift scenarios"""
    
    print("🔧 TESTING SEMANTIC DRIFT FIXES")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Testing context anchoring for ambiguous follow-ups")
    
    api_key = "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"
    tester = SemanticDriftTester(api_key)
    
    results = []
    
    # Test 1: The original failing scenario - stimulant fatigue discussion
    print("\\n" + "="*60)
    print("🧪 TEST 1: Original Failing Scenario (Stimulant Fatigue)")
    print("="*60)
    
    thread_id = "stimulant_fatigue_test"
    
    print("💬 Step 1: Set up stimulant fatigue context")
    response1, _ = tester.chat(
        "I've been dealing with some fatigue issues, likely from the stimulants I'm on. Not sure if it's because I need more sleep or more exercise.",
        thread_id
    )
    print(f"🤖 {response1}")
    
    print("\\n💬 Step 2: Assistant follow-up question")
    response2, _ = tester.chat(
        "Have you noticed any changes in your routine?",
        thread_id
    )
    print(f"🤖 {response2}")
    
    print("\\n💬 Step 3: CRITICAL - Ambiguous 'what do you think' follow-up")
    response3, messages3 = tester.chat("what do you think", thread_id)
    print(f"🤖 {response3}")
    
    # Check if response stayed anchored to fatigue/stimulant context
    fatigue_context_maintained = any(word in response3.lower() for word in [
        "fatigue", "stimulant", "sleep", "exercise", "routine", "energy", "tired"
    ])
    
    generic_ai_response = any(phrase in response3.lower() for phrase in [
        "different perspectives", "depends on what", "could mean many things", 
        "various ways to interpret", "hard to say without", "it really depends"
    ])
    
    if fatigue_context_maintained and not generic_ai_response:
        print("✅ SUCCESS: Context anchored to fatigue discussion")
        results.append(("Fatigue Context Anchoring", True))
    else:
        print("❌ FAILED: Drifted into generic AI response")
        results.append(("Fatigue Context Anchoring", False))
    
    # Test 2: Technology discussion with "sure" response
    print("\\n" + "="*60) 
    print("🧪 TEST 2: Technology Discussion with 'Sure' Response")
    print("="*60)
    
    thread_id2 = "tech_discussion_test"
    
    print("💬 Step 1: Technical discussion setup")
    response4, _ = tester.chat(
        "I'm debating between React and Vue for the frontend. React has more ecosystem support but Vue is simpler.",
        thread_id2
    )
    print(f"🤖 {response4}")
    
    print("\\n💬 Step 2: CRITICAL - Ambiguous 'sure' response")
    response5, messages5 = tester.chat("sure", thread_id2)
    print(f"🤖 {response5}")
    
    # Check if response stayed anchored to React/Vue context
    tech_context_maintained = any(word in response5.lower() for word in [
        "react", "vue", "frontend", "ecosystem", "framework", "javascript", "component"
    ])
    
    if tech_context_maintained:
        print("✅ SUCCESS: Context anchored to React/Vue discussion")
        results.append(("Tech Context Anchoring", True))
    else:
        print("❌ FAILED: Lost React/Vue context")
        results.append(("Tech Context Anchoring", False))
    
    # Test 3: Complex conversation with "depends" response
    print("\\n" + "="*60)
    print("🧪 TEST 3: Complex Discussion with 'Depends' Response")
    print("="*60)
    
    thread_id3 = "complex_discussion_test"
    
    print("💬 Step 1: Complex scenario setup")
    response6, _ = tester.chat(
        "I'm thinking about optimizing our database queries. We're seeing slow performance on user lookups, especially with large datasets.",
        thread_id3
    )
    print(f"🤖 {response6}")
    
    print("\\n💬 Step 2: Follow-up question")
    response7, _ = tester.chat(
        "Would indexing help or should we consider query restructuring?",
        thread_id3
    )
    print(f"🤖 {response7}")
    
    print("\\n💬 Step 3: CRITICAL - Ambiguous 'depends' response")
    response8, messages8 = tester.chat("depends", thread_id3)
    print(f"🤖 {response8}")
    
    # Check if response stayed anchored to database optimization context
    db_context_maintained = any(word in response8.lower() for word in [
        "database", "queries", "performance", "indexing", "optimization", "lookup", "restructur"
    ])
    
    if db_context_maintained:
        print("✅ SUCCESS: Context anchored to database discussion")
        results.append(("Database Context Anchoring", True))
    else:
        print("❌ FAILED: Lost database context")
        results.append(("Database Context Anchoring", False))
    
    # Test 4: Pattern detection accuracy
    print("\\n" + "="*60)
    print("🧪 TEST 4: Pattern Detection Accuracy")
    print("="*60)
    
    test_phrases = [
        ("what do you think", True),
        ("yeah", True),
        ("sure", True),
        ("depends", True),
        ("that makes sense", True),
        ("I think React is better", False),
        ("Let's discuss Vue instead", False),
        ("What about database performance?", True),
        ("The indexing approach seems good", False),
    ]
    
    pattern_accuracy = 0
    for phrase, should_match in test_phrases:
        detected = tester._is_ambiguous_followup(phrase)
        if detected == should_match:
            pattern_accuracy += 1
            print(f"✅ '{phrase}' - Correctly {'detected' if should_match else 'ignored'}")
        else:
            print(f"❌ '{phrase}' - {'Missed detection' if should_match else 'False positive'}")
    
    pattern_test_passed = pattern_accuracy >= len(test_phrases) * 0.8  # 80% accuracy
    results.append(("Pattern Detection", pattern_test_passed))
    
    # Results summary
    print("\\n" + "="*60)
    print("📊 SEMANTIC DRIFT FIX RESULTS")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\\n🎯 Fix Score: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL SEMANTIC DRIFT FIXES WORKING!")
        print("🔗 Context anchoring prevents generic AI responses")
        print("🧠 Ambiguous follow-ups stay tied to specific topics")
    elif passed >= 3:
        print("✅ MAJOR IMPROVEMENT! Most semantic drift issues resolved")
    else:
        print("⚠️  NEEDS MORE WORK - Semantic drift still occurring")


if __name__ == "__main__":
    test_semantic_drift_fixes()
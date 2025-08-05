"""
Direct OpenAI Integration - GPT-4o Optimized Conversation Management
Human-like responses with sophisticated memory integration and context awareness
"""
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import json
import re
from datetime import datetime
from core.memory_engine import Memory, MemoryEngine
from core.logging_config import get_logger, monitor_performance
from .embeddings import OpenAIEmbeddings
import os


class DirectOpenAIChat:
    """Direct OpenAI chat integration with full control over conversation context"""
    
    def __init__(
        self,
        api_key: str,
        memory_engine: MemoryEngine,
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-ada-002",
        system_prompt_path: str = "./prompts/system_prompt_4o.txt",
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.memory_engine = memory_engine
        self.embeddings = OpenAIEmbeddings(api_key, embedding_model)
        self.logger = get_logger("direct_openai")
        self.system_prompt_path = system_prompt_path
        
        # Store conversation history in memory by thread_id
        self.conversations = {}
        self._load_conversation_history()
        
        # Load optimized system prompt
        self.base_system_prompt = self._load_system_prompt()
        
        # User identity profile for GPT-4o personalization
        self.user_identity = {
            "name": "Jeremy Kimble",
            "role": "IT consultant and AI system builder",
            "preferences": "Direct communication, technical precision, no fluff responses",
            "context": "Building AI memory layer with vector storage and GPT-4o integration",
            "communication_style": "Expects blunt, helpful responses like a capable peer"
        }
        
        # Ambiguous follow-up patterns that need context anchoring
        self.ambiguous_patterns = [
            r"^what (about|do you think|should I do)",
            r"^yeah(?!\w)",
            r"^sure(?!\w)",
            r"^ok(?!\w)",
            r"^maybe(?!\w)",
            r"^probably(?!\w)",
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
            r"^no(?!\w)",
            r"^yes(?!\w)",
            r"^I don't know",
            r"^not sure",
            r"^you're right",
            r"^maybe not",
        ]
        
    def _load_conversation_history(self):
        """Load persisted conversation history from disk"""
        history_path = os.path.join(
            os.environ.get('PERSIST_DIRECTORY', './data'),
            'conversation_history.json'
        )
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    self.conversations = json.load(f)
                self.logger.info(f"Loaded {len(self.conversations)} conversation threads")
            except Exception as e:
                self.logger.error(f"Failed to load conversation history: {e}")
                self.conversations = {}
    
    def _save_conversation_history(self):
        """Persist conversation history to disk"""
        history_path = os.path.join(
            os.environ.get('PERSIST_DIRECTORY', './data'),
            'conversation_history.json'
        )
        try:
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            with open(history_path, 'w') as f:
                json.dump(self.conversations, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save conversation history: {e}")
    
    def _load_system_prompt(self) -> str:
        """Load the GPT-4o optimized system prompt"""
        try:
            if os.path.exists(self.system_prompt_path):
                with open(self.system_prompt_path, 'r') as f:
                    return f.read().strip()
        except Exception as e:
            self.logger.error(f"Failed to load system prompt: {e}")
        
        # Fallback to embedded prompt
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

If retrieved memory is provided, treat it as background insight. Use it to inform your response, not dominate it. Only reference it directly if it clarifies the user's intent.

You are currently supporting a user named Jeremy Kimble, an IT consultant who is building a long-memory AI assistant with vector recall using FAISS and OpenAI's GPT-4o API. He values speed, precision, low-fluff responses, and clever utility."""
    
    def _dedupe_and_paraphrase_memories(self, contents: List[str], memory_type: str) -> List[str]:
        """Deduplicate and paraphrase memory content to avoid redundancy"""
        if not contents:
            return []
        
        # Simple semantic deduplication based on key terms
        unique_contents = []
        seen_keywords = set()
        
        for content in contents:
            # Extract key terms for similarity checking
            key_terms = set(word.lower() for word in content.split() 
                           if len(word) > 3 and word.isalpha())
            
            # Check if content is too similar to existing ones
            overlap = any(len(key_terms & seen) > 2 for seen in seen_keywords)
            
            if not overlap:
                unique_contents.append(content)
                seen_keywords.add(frozenset(key_terms))
            
            # Limit to prevent overwhelming context
            if len(unique_contents) >= 3:
                break
        
        return unique_contents
    
    def _create_memory_summary(self, memories: List[Memory]) -> str:
        """Transform raw FAISS memories into natural context summaries with deduplication"""
        if not memories:
            return None
            
        # Group memories by type and create natural summaries
        memory_groups = {}
        for memory in memories:
            memory_type = memory.metadata.get('type', 'general')
            if memory_type not in memory_groups:
                memory_groups[memory_type] = []
            memory_groups[memory_type].append(memory.content)
        
        summaries = []
        for mem_type, contents in memory_groups.items():
            # Deduplicate and paraphrase based on type
            unique_contents = self._dedupe_and_paraphrase_memories(contents, mem_type)
            
            if mem_type == 'preference' and unique_contents:
                # Vary phrasing for preferences to avoid repetition
                pref_phrases = [
                    f"Jeremy values {', '.join(unique_contents[:2])}",
                    f"Communication style: {', '.join(unique_contents[:2])}",
                    f"Prefers {', '.join(unique_contents[:2])}"
                ]
                summaries.append(pref_phrases[len(summaries) % len(pref_phrases)])
            elif mem_type == 'user_message' and unique_contents:
                summaries.append(f"Previous topics: {', '.join([c[:50] + '...' for c in unique_contents[:2]])}")
            elif (mem_type == 'tool' or mem_type == 'technical') and unique_contents:
                summaries.append(f"Technical background: {', '.join(unique_contents[:2])}")
            elif unique_contents:
                summaries.append(f"Context: {unique_contents[0][:100]}...")
        
        return " | ".join(summaries) if summaries else None
    
    def _create_identity_message(self, thread_id: str = None) -> str:
        """Create identity profile message with explicit name/style anchoring"""
        # Check if this is early in the thread for stronger identity anchoring
        is_thread_start = False
        if thread_id and thread_id in self.conversations:
            is_thread_start = len(self.conversations[thread_id]) < 4
        
        if is_thread_start or not thread_id:
            # Stronger identity framing for thread openings
            return f"""User Identity: You're speaking with {self.user_identity['name']}, who explicitly prefers {self.user_identity['communication_style'].lower()}. Jeremy dislikes robotic responses, values technical precision, and expects you to reference conversation history naturally. Current focus: {self.user_identity['context']}."""
        else:
            # Standard profile for mid-thread
            return f"""User Profile: {self.user_identity['name']}, {self.user_identity['role']}. Communication style: {self.user_identity['communication_style']}. Current project context: {self.user_identity['context']}."""
    
    def _create_behavior_log(self) -> str:
        """Create past behavior expectations for consistency"""
        return """Past interaction patterns: The assistant is expected to avoid caveats, provide formatted code when asked, speak like a capable peer (not customer support), reference conversation history naturally, and maintain continuity across topics without restarting context. CRITICAL: Short responses or single words are usually replies to previous questions, not new topics or name declarations."""
    
    def _get_conversation_messages(self, thread_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation messages for a thread - ALWAYS maintain at least 3-5 exchanges"""
        if thread_id not in self.conversations:
            self.conversations[thread_id] = []
        
        # Ensure we keep at least 6 messages (3 user-assistant exchanges) minimum
        effective_limit = max(limit, 6) if limit > 0 else 0
        
        # Return the last N messages (both user and assistant)
        return self.conversations[thread_id][-effective_limit:] if effective_limit > 0 else self.conversations[thread_id]
    
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
        
        self._save_conversation_history()
    
    def _classify_input_type(self, user_input: str, thread_id: str) -> str:
        """Classify user input to distinguish replies from new topics"""
        # Get recent conversation to check context
        recent_messages = self._get_conversation_messages(thread_id, limit=4)
        
        # If input is very short and follows an assistant question, it's likely a reply
        if len(user_input.split()) <= 3 and recent_messages:
            # Check if the last assistant message ended with a question
            last_assistant_msg = None
            for msg in reversed(recent_messages):
                if msg["role"] == "assistant":
                    last_assistant_msg = msg["content"]
                    break
            
            if last_assistant_msg and ("?" in last_assistant_msg or 
                                     "what" in last_assistant_msg.lower() or
                                     "which" in last_assistant_msg.lower() or
                                     "how" in last_assistant_msg.lower()):
                return "reply"
        
        # Check for common greeting patterns
        greeting_patterns = ["hello", "hi", "hey", "good morning", "good afternoon"]
        if any(pattern in user_input.lower() for pattern in greeting_patterns):
            return "greeting"
        
        # Check for meta/system references
        meta_patterns = ["claude", "ai", "assistant", "system", "model"]
        if any(pattern in user_input.lower() for pattern in meta_patterns) and len(user_input.split()) < 5:
            return "meta"
        
        # Default to topic for longer messages
        return "topic"
    
    def _disambiguate_user_input(self, user_input: str, thread_id: str) -> str:
        """Disambiguate user input based on conversational context"""
        input_type = self._classify_input_type(user_input, thread_id)
        
        if input_type == "reply":
            return f"User replied: {user_input}"
        elif input_type == "meta" and "claude" in user_input.lower():
            # Special handling for "Claude Code" type responses
            return f"User answered with: {user_input} (referring to what they're working on, not their name)"
        
        # Return original input for greetings and topics
        return user_input
    
    def _is_ambiguous_followup(self, user_input: str) -> bool:
        """Detect if user input is an ambiguous follow-up that needs context anchoring"""
        user_lower = user_input.lower().strip()
        
        # Check against ambiguous patterns
        for pattern in self.ambiguous_patterns:
            if re.match(pattern, user_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _get_last_assistant_message(self, thread_id: str) -> Optional[str]:
        """Get the most recent assistant message from conversation history"""
        recent_messages = self._get_conversation_messages(thread_id, limit=10)
        
        # Find the last assistant message
        for msg in reversed(recent_messages):
            if msg["role"] == "assistant":
                return msg["content"]
        
        return None
    
    def _create_context_anchor(self, user_input: str, thread_id: str) -> Optional[str]:
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
    
    def _build_messages_array(
        self,
        thread_id: str,
        user_message: str,
        system_prompt: Optional[str] = None,
        include_memories: int = 5
    ) -> List[Dict[str, str]]:
        """Build the optimized messages array for GPT-4o API with sophisticated context"""
        messages = []
        
        # 1. Primary system prompt (GPT-4o optimized)
        if not system_prompt:
            system_prompt = self.base_system_prompt
        messages.append({"role": "system", "content": system_prompt})
        
        # 2. Identity profile injection for personalization (enhanced for thread starts)
        messages.append({"role": "system", "content": self._create_identity_message(thread_id)})
        
        # 3. Behavior expectations for consistency
        messages.append({"role": "system", "content": self._create_behavior_log()})
        
        # 4. High-priority memories (identity corrections, etc.) - ALWAYS include
        if hasattr(self.memory_engine, 'get_high_priority_memories'):
            high_priority_memories = self.memory_engine.get_high_priority_memories(limit=3)
            if high_priority_memories:
                priority_content = " | ".join([mem.content for mem in high_priority_memories])
                messages.append({
                    "role": "system",
                    "content": f"CRITICAL CONTEXT (always apply): {priority_content}"
                })
        
        # 5. Sophisticated memory injection (rewritten for natural context)
        if include_memories > 0:
            relevant_memories = self.memory_engine.search_memories(user_message, k=include_memories)
            if relevant_memories:
                memory_summary = self._create_memory_summary(relevant_memories)
                if memory_summary:
                    messages.append({
                        "role": "system", 
                        "content": f"Background context: {memory_summary}"
                    })
        
        # 6. Context anchoring for ambiguous follow-ups (CRITICAL for semantic drift prevention)
        context_anchor = self._create_context_anchor(user_message, thread_id)
        if context_anchor:
            messages.append({
                "role": "system",
                "content": context_anchor
            })
            self.logger.debug(f"Added context anchor for ambiguous input: {user_message[:50]}...")
        
        # 7. Add conversation history (maintain continuity - minimum 3-5 exchanges)
        conversation_history = self._get_conversation_messages(thread_id, limit=20)
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 8. Add current user message with disambiguation
        disambiguated_message = self._disambiguate_user_input(user_message, thread_id)
        messages.append({"role": "user", "content": disambiguated_message})
        
        self.logger.debug(
            f"Built GPT-4o optimized message array with {len(messages)} messages for thread {thread_id}",
            extra={
                "system_messages": sum(1 for m in messages if m["role"] == "system"),
                "history_messages": len(conversation_history),
                "total_messages": len(messages),
                "memory_injected": include_memories > 0,
                "context_anchored": context_anchor is not None,
                "ambiguous_input": self._is_ambiguous_followup(user_message)
            }
        )
        
        return messages
    
    @monitor_performance("gpt4o_chat_completion")
    def chat(
        self,
        message: str,
        thread_id: str = "default",
        system_prompt: Optional[str] = None,
        remember_response: bool = True,
        temperature: float = 0.7,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """GPT-4o optimized chat method with human-like response tuning"""
        
        self.logger.info(
            f"Processing chat request for thread {thread_id}",
            extra={"message_length": len(message), "thread_id": thread_id}
        )
        
        # Build messages array with full context
        messages = self._build_messages_array(
            thread_id=thread_id,
            user_message=message,
            system_prompt=system_prompt
        )
        
        try:
            # GPT-4o optimized API call with human-like tuning
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,  # 0.6-0.75 for human-like responses
                top_p=1.0,  # Full token diversity
                presence_penalty=0.5,  # Encourage new topics/ideas
                frequency_penalty=0.25,  # Reduce repetition
                max_tokens=1200  # Allow for more detailed responses
            )
            
            assistant_response = response.choices[0].message.content
            
            self.logger.info(
                f"Received response from OpenAI",
                extra={
                    "response_length": len(assistant_response),
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                    "thread_id": thread_id
                }
            )
            
            # Store in conversation history
            if remember_response:
                self._add_to_conversation(thread_id, "user", message)
                self._add_to_conversation(thread_id, "assistant", assistant_response)
                
                # Detect and store identity corrections
                self._detect_and_store_corrections(message, assistant_response, thread_id)
                
                # Also store in long-term memory for search
                self.memory_engine.add_memory(
                    f"User: {message}",
                    metadata={"type": "user_message", "thread_id": thread_id}
                )
                self.memory_engine.add_memory(
                    f"Assistant: {assistant_response}",
                    metadata={"type": "assistant_response", "thread_id": thread_id}
                )
            
            return assistant_response, messages
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}", exc_info=True)
            raise
    
    def get_conversation_history(self, thread_id: str) -> List[Dict[str, str]]:
        """Get full conversation history for a thread"""
        return self._get_conversation_messages(thread_id, limit=0)
    
    def clear_thread(self, thread_id: str):
        """Clear conversation history for a specific thread"""
        if thread_id in self.conversations:
            del self.conversations[thread_id]
            self._save_conversation_history()
            self.logger.info(f"Cleared conversation thread {thread_id}")
    
    def get_thread_summary(self, thread_id: str) -> str:
        """Generate a summary of a conversation thread"""
        messages = self.get_conversation_history(thread_id)
        if not messages:
            return "No conversation history"
        
        # Extract just the content for summary
        conversation_text = "\n".join([
            f"{msg['role'].title()}: {msg['content']}" 
            for msg in messages[-10:]  # Last 10 messages
        ])
        
        return conversation_text[:500] + "..." if len(conversation_text) > 500 else conversation_text
    
    def _detect_and_store_corrections(self, user_message: str, assistant_response: str, thread_id: str):
        """Detect identity corrections and name clarifications from user messages"""
        user_lower = user_message.lower()
        assistant_lower = assistant_response.lower()
        
        # Detect name corrections
        name_correction_patterns = [
            "my name isn't", "my name is not", "i'm not", "don't call me",
            "not my name", "that's not my name", "wrong name"
        ]
        
        # Detect identity corrections
        if any(pattern in user_lower for pattern in name_correction_patterns):
            if hasattr(self.memory_engine, 'add_identity_correction'):
                self.memory_engine.add_identity_correction(
                    f"User corrected name/identity: {user_message}",
                    f"Context: {assistant_response[:100]}..."
                )
                self.logger.info(f"Stored identity correction for thread {thread_id}")
        
        # Detect "Claude Code" type clarifications
        if ("claude" in user_lower and len(user_message.split()) <= 3 and 
            "what" in assistant_lower and "working" in assistant_lower):
            if hasattr(self.memory_engine, 'add_identity_correction'):
                self.memory_engine.add_identity_correction(
                    f"When user says '{user_message}', they're answering what they're working on, not stating their name",
                    f"Context: Previous question about work/projects"
                )
                self.logger.info(f"Stored work context clarification for thread {thread_id}")
        
        # Detect meta-conversation corrections
        if "you actually" in user_lower or "you were" in user_lower:
            recent_history = self._get_conversation_messages(thread_id, limit=6)
            if len(recent_history) >= 2:
                context = f"User correcting assistant behavior: {user_message}"
                if hasattr(self.memory_engine, 'add_identity_correction'):
                    self.memory_engine.add_identity_correction(
                        context,
                        f"Assistant should maintain conversation continuity better"
                    )
                    self.logger.info(f"Stored conversation flow correction for thread {thread_id}")
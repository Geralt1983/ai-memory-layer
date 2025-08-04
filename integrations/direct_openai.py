"""
Direct OpenAI Integration - GPT-4o Optimized Conversation Management
Human-like responses with sophisticated memory integration and context awareness
"""
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import json
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
    
    def _create_memory_summary(self, memories: List[Memory]) -> str:
        """Transform raw FAISS memories into natural context summaries"""
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
            if mem_type == 'preference':
                summaries.append(f"User preferences: {', '.join(contents[:3])}")
            elif mem_type == 'user_message':
                summaries.append(f"Recent topics discussed: {', '.join([c[:50] + '...' for c in contents[:2]])}")
            elif mem_type == 'tool' or mem_type == 'technical':
                summaries.append(f"Technical context: {', '.join(contents[:2])}")
            else:
                summaries.append(f"Context: {contents[0][:100]}..." if contents else "")
        
        return " | ".join(summaries) if summaries else None
    
    def _create_identity_message(self) -> str:
        """Create identity profile message for GPT-4o personalization"""
        return f"""User Profile: {self.user_identity['name']}, {self.user_identity['role']}. Communication style: {self.user_identity['communication_style']}. Current project context: {self.user_identity['context']}."""
    
    def _create_behavior_log(self) -> str:
        """Create past behavior expectations for consistency"""
        return """Past interaction patterns: The assistant is expected to avoid caveats, provide formatted code when asked, speak like a capable peer (not customer support), reference conversation history naturally, and maintain continuity across topics without restarting context."""
    
    def _get_conversation_messages(self, thread_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation messages for a thread"""
        if thread_id not in self.conversations:
            self.conversations[thread_id] = []
        
        # Return the last N messages (both user and assistant)
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
        
        self._save_conversation_history()
    
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
        
        # 2. Identity profile injection for personalization
        messages.append({"role": "system", "content": self._create_identity_message()})
        
        # 3. Behavior expectations for consistency
        messages.append({"role": "system", "content": self._create_behavior_log()})
        
        # 4. Sophisticated memory injection (rewritten for natural context)
        if include_memories > 0:
            relevant_memories = self.memory_engine.search_memories(user_message, k=include_memories)
            if relevant_memories:
                memory_summary = self._create_memory_summary(relevant_memories)
                if memory_summary:
                    messages.append({
                        "role": "system", 
                        "content": f"Background context: {memory_summary}"
                    })
        
        # 5. Add conversation history (maintain continuity)
        conversation_history = self._get_conversation_messages(thread_id, limit=20)
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 6. Add current user message
        messages.append({"role": "user", "content": user_message})
        
        self.logger.debug(
            f"Built GPT-4o optimized message array with {len(messages)} messages for thread {thread_id}",
            extra={
                "system_messages": sum(1 for m in messages if m["role"] == "system"),
                "history_messages": len(conversation_history),
                "total_messages": len(messages),
                "memory_injected": include_memories > 0
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
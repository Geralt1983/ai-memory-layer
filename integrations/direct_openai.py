"""
Direct OpenAI Integration - Clean ChatGPT-like conversation management
No LangChain, No LangGraph, Just direct control over the conversation flow
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
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.memory_engine = memory_engine
        self.embeddings = OpenAIEmbeddings(api_key, embedding_model)
        self.logger = get_logger("direct_openai")
        
        # Store conversation history in memory by thread_id
        self.conversations = {}
        self._load_conversation_history()
        
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
        
        # 2. Add relevant memories from long-term storage
        if include_memories > 0:
            relevant_memories = self.memory_engine.search_memories(user_message, k=include_memories)
            if relevant_memories:
                memory_context = "\n".join([
                    f"- {mem.content}" for mem in relevant_memories
                ])
                messages.append({
                    "role": "system", 
                    "content": f"Relevant context from memory:\n{memory_context}"
                })
        
        # 3. Add conversation history (this is the KEY part!)
        conversation_history = self._get_conversation_messages(thread_id, limit=20)
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 4. Add current user message
        messages.append({"role": "user", "content": user_message})
        
        self.logger.debug(
            f"Built message array with {len(messages)} messages for thread {thread_id}",
            extra={
                "system_messages": sum(1 for m in messages if m["role"] == "system"),
                "history_messages": len(conversation_history),
                "total_messages": len(messages)
            }
        )
        
        return messages
    
    @monitor_performance("chat_completion")
    def chat(
        self,
        message: str,
        thread_id: str = "default",
        system_prompt: Optional[str] = None,
        remember_response: bool = True,
        temperature: float = 0.7,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Main chat method - returns response and the context used"""
        
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
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000
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
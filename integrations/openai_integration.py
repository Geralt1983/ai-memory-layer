from typing import List, Dict, Any, Optional
from openai import OpenAI
import numpy as np
from core.memory_engine import Memory, MemoryEngine
from core.context_builder import ContextBuilder
from core.conversation_buffer import ConversationBuffer
from core.logging_config import get_logger, monitor_performance
from .embeddings import OpenAIEmbeddings


class OpenAIIntegration:
    def __init__(
        self,
        api_key: str,
        memory_engine: MemoryEngine,
        model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
        conversation_buffer_size: int = 20,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.memory_engine = memory_engine
        self.embeddings = OpenAIEmbeddings(api_key, embedding_model)
        self.context_builder = ContextBuilder(memory_engine)
        self.conversation_buffer = ConversationBuffer(max_messages=conversation_buffer_size)
        self.logger = get_logger("openai_integration")

        self.logger.info(
            "OpenAI integration initialized",
            extra={"model": model, "embedding_model": embedding_model},
        )

    def _extract_user_preferences(self) -> str:
        """Extract user preferences and behavior guidance from memory"""
        # Search for preference-related memories
        preference_queries = ["prefer", "like", "want you to", "style", "tone", "avoid", "don't"]
        preferences = []
        
        for query in preference_queries:
            memories = self.memory_engine.search_memories(query, k=3)
            for memory in memories:
                if memory.relevance_score > 0.6:  # Only include relevant preferences
                    # Check if it's a user message (preferences are usually from user)
                    if (memory.metadata.get("type") == "user_message" or 
                        "User:" in memory.content):
                        preferences.append(memory.content)
        
        if preferences:
            return "\n".join(set(preferences))  # Remove duplicates
        return ""
    
    def _build_system_prompt(self, user_preferences: str, context: str, conversation_context: str = "") -> str:
        """Build an enhanced system prompt with user preferences, long-term context, and recent conversation"""
        base_prompt = """You are an AI assistant with persistent memory. You have both long-term memory (past conversations and knowledge) and short-term memory (recent conversation context). MUST adapt your responses based on the user's explicitly stated preferences and feedback."""
        
        prompt_parts = [base_prompt]
        
        if user_preferences:
            prompt_parts.append(f"""
## CRITICAL USER PREFERENCES - MUST FOLLOW:
{user_preferences}

⚠️ MANDATORY: The user has explicitly told you how they want you to communicate. You MUST follow these preferences strictly:
- If they want you to be "direct" - be direct, not verbose
- If they want you to be "less cheery" - be more neutral/serious
- If they want "human-like" responses - avoid AI-like formatting and be conversational
- If they said "avoid generic answers" - give specific, personalized responses
- If they called responses "canned" - be more natural and spontaneous

IGNORE your default response style and adapt completely to their stated preferences.""")
        
        if conversation_context:
            prompt_parts.append(f"""
## Recent Conversation Context:
{conversation_context}

This is your recent conversation with the user. Use this to maintain context and avoid repeating questions or information.""")
        
        if context:
            prompt_parts.append(f"""
## Relevant Long-term Memory:
{context}

This is relevant information from past conversations and stored knowledge. Use this to personalize your response.""")
        
        prompt_parts.append("""
## RESPONSE REQUIREMENTS:
1. FIRST check user preferences above and adapt your entire response style accordingly
2. Reference the user's past conversations and stated preferences
3. Be authentic and avoid generic, templated responses  
4. If the user has corrected your behavior before, remember and apply those corrections
5. Make your response feel natural and personalized to this specific user
6. CRITICAL: If you already know information about the user from past conversations, USE that knowledge instead of asking again
7. Don't ask questions about things you already know (like asking for names you already have)
8. Build on existing knowledge rather than starting over

Remember: The user has a memory system specifically so you can learn their preferences and adapt. USE IT INTELLIGENTLY.""")
        
        return "\n".join(prompt_parts)

    @monitor_performance("chat_with_memory")
    def chat_with_memory(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        include_recent: int = 5,
        include_relevant: int = 5,
        remember_response: bool = True,
    ) -> str:
        self.logger.info(
            "Starting chat with memory",
            extra={
                "message_length": len(message),
                "include_recent": include_recent,
                "include_relevant": include_relevant,
                "remember_response": remember_response,
                "has_system_prompt": system_prompt is not None,
            },
        )

        # Extract user preferences from memory
        self.logger.debug("Extracting user preferences from memory")
        user_preferences = self._extract_user_preferences()
        
        # Build context from long-term memory
        self.logger.debug("Building context from long-term memory")
        context = self.context_builder.build_context(
            query=message,
            include_recent=include_recent,
            include_relevant=include_relevant,
        )
        
        # Get recent conversation context
        conversation_context = self.conversation_buffer.get_context_string(max_chars=1500)

        # Prepare messages with enhanced system prompt
        messages = []
        
        if system_prompt:
            # Use provided system prompt
            messages.append({"role": "system", "content": system_prompt})
        else:
            # Build enhanced system prompt with preferences, context, and conversation
            enhanced_system_prompt = self._build_system_prompt(user_preferences, context, conversation_context)
            messages.append({"role": "system", "content": enhanced_system_prompt})
            self.logger.debug(
                "Enhanced system prompt created", 
                extra={
                    "has_preferences": bool(user_preferences),
                    "context_length": len(context) if context else 0,
                    "conversation_messages": self.conversation_buffer.get_message_count()
                }
            )

        # Add recent conversation messages for immediate context
        recent_messages = self.conversation_buffer.get_messages()
        messages.extend(recent_messages)
        
        # Add current user message
        messages.append({"role": "user", "content": message})

        # Get response from OpenAI
        self.logger.debug(
            "Sending request to OpenAI",
            extra={"model": self.model, "message_count": len(messages)},
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages
            )

            answer = response.choices[0].message.content

            self.logger.info(
                "OpenAI response received",
                extra={
                    "response_length": len(answer) if answer else 0,
                    "model": self.model,
                },
            )

        except Exception as e:
            self.logger.error(
                "Failed to get OpenAI response",
                extra={"error": str(e), "model": self.model},
                exc_info=True,
            )
            raise

        # Store the interaction in memory
        if remember_response:
            self.logger.debug("Storing conversation in memory")

            # Add to conversation buffer (short-term memory)
            self.conversation_buffer.add_message("user", message)
            self.conversation_buffer.add_message("assistant", answer)

            # Store user message (MemoryEngine will handle embedding generation)
            self.memory_engine.add_memory(
                f"User: {message}", metadata={"type": "user_message"}
            )

            # Store assistant response (MemoryEngine will handle embedding generation)
            self.memory_engine.add_memory(
                f"Assistant: {answer}", metadata={"type": "assistant_response"}
            )

            self.logger.debug("Conversation stored in both buffer and persistent memory successfully")

        self.logger.info(
            "Chat with memory completed successfully",
            extra={
                "message_length": len(message),
                "response_length": len(answer) if answer else 0,
            },
        )

        return answer

    def add_memory_with_embedding(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Memory:
        # MemoryEngine now handles embedding generation internally
        return self.memory_engine.add_memory(content, metadata)
    
    def clear_conversation_buffer(self) -> None:
        """Clear the conversation buffer (start fresh conversation)."""
        self.conversation_buffer.clear()
        self.logger.info("Conversation buffer cleared")
    
    def get_conversation_buffer_info(self) -> Dict[str, Any]:
        """Get information about the current conversation buffer."""
        return {
            "message_count": self.conversation_buffer.get_message_count(),
            "max_messages": self.conversation_buffer.max_messages,
            "recent_context": self.conversation_buffer.get_context_string(max_chars=500)
        }

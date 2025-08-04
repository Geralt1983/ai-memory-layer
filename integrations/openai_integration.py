from typing import List, Dict, Any, Optional
from openai import OpenAI
import numpy as np
from core.memory_engine import Memory, MemoryEngine
from core.context_builder import ContextBuilder
from core.conversation_buffer import ConversationBuffer
from core.logging_config import get_logger, monitor_performance
from .embeddings import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain


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
        
        # Use LangChain's full conversation system instead of raw OpenAI API
        self.langchain_chat = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model,
            temperature=0.7
        )
        
        self.langchain_memory = ConversationSummaryBufferMemory(
            llm=self.langchain_chat,
            max_token_limit=4000,  # Increased from 2000 - keep more context
            return_messages=False,  # Return as string for ConversationChain
            memory_key="history"
        )
        
        # This is the key - use LangChain's ConversationChain which handles flow properly
        self.conversation_chain = ConversationChain(
            llm=self.langchain_chat,
            memory=self.langchain_memory,
            verbose=True  # Enable to debug what LangChain is actually doing
        )
        
        # Keep custom buffer for compatibility (for now)
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
    
    def _build_system_prompt(self, user_preferences: str, context: str) -> str:
        """Build a natural system prompt that encourages proper conversation flow"""
        prompt = f"""You're Jeremy's AI assistant. Pay close attention to conversation flow and context.

About Jeremy: 41 years old, wife Ashley, 7 kids, dogs Remy & Bailey. Direct communicator who dislikes generic responses.

{f"Long-term context: {context}" if context else ""}
{f"His preferences: {user_preferences}" if user_preferences else ""}

Key: Always acknowledge and build on Jeremy's specific answers. When he asks "what do you think" or similar, refer to what was just discussed."""
        
        return prompt

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
            "Starting chat with LangChain conversation chain",
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
        
        # Let LangChain handle conversation naturally without custom prompt injection
        # Only inject system context if explicitly provided
        if system_prompt:
            from langchain.prompts import PromptTemplate
            custom_prompt = PromptTemplate(
                input_variables=["history", "input"],
                template=f"""{system_prompt}

Current conversation:
{{history}}
Human: {{input}}
Assistant:"""
            )
            self.conversation_chain.prompt = custom_prompt
        
        self.logger.debug(
            "Using LangChain ConversationChain", 
            extra={
                "has_custom_prompt": system_prompt is not None,
                "has_preferences": bool(user_preferences),
                "context_length": len(context) if context else 0,
                "chain_memory_messages": len(self.langchain_memory.chat_memory.messages),
                "chain_max_tokens": self.langchain_memory.max_token_limit
            }
        )

        try:
            # Debug: Log what's actually in the memory before making the call
            memory_content = self.langchain_memory.buffer
            chat_messages = self.langchain_memory.chat_memory.messages
            
            self.logger.info(
                "ConversationSummaryBufferMemory Debug",
                extra={
                    "memory_buffer": memory_content[:1000] + "..." if len(memory_content) > 1000 else memory_content,
                    "buffer_length": len(memory_content),
                    "chat_messages_count": len(chat_messages),
                    "max_token_limit": self.langchain_memory.max_token_limit,
                    "current_message": message
                }
            )
            
            # Debug: Log the last few chat messages to see what's being kept
            if chat_messages:
                recent_messages = []
                for msg in chat_messages[-6:]:  # Last 3 exchanges (6 messages)
                    recent_messages.append(f"{msg.__class__.__name__}: {msg.content}")
                
                self.logger.info(
                    "Recent chat messages in memory",
                    extra={"recent_messages": recent_messages}
                )
            
            # Use LangChain's conversation chain - this handles conversation flow properly!
            answer = self.conversation_chain.predict(input=message)

            self.logger.info(
                "LangChain conversation response received",
                extra={
                    "response_length": len(answer) if answer else 0,
                    "model": self.model,
                },
            )

        except Exception as e:
            self.logger.error(
                "Failed to get LangChain conversation response",
                extra={"error": str(e), "model": self.model},
                exc_info=True,
            )
            raise

        # Store the interaction in persistent memory (LangChain memory is handled automatically)
        if remember_response:
            self.logger.debug("Storing conversation in persistent memory")

            # Add to custom buffer for compatibility (for now)
            self.conversation_buffer.add_message("user", message)
            self.conversation_buffer.add_message("assistant", answer)

            # Store user message in persistent memory (MemoryEngine will handle embedding generation)
            self.memory_engine.add_memory(
                f"User: {message}", metadata={"type": "user_message"}
            )

            # Store assistant response in persistent memory
            self.memory_engine.add_memory(
                f"Assistant: {answer}", metadata={"type": "assistant_response"}
            )

            self.logger.debug("Conversation stored in persistent memory successfully")

        self.logger.info(
            "LangChain chat with memory completed successfully",
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
        """Clear LangChain conversation chain memory and custom conversation buffer (start fresh conversation)."""
        self.conversation_chain.memory.clear()
        self.conversation_buffer.clear()
        self.logger.info("LangChain conversation chain memory and conversation buffer cleared")
    
    def get_conversation_buffer_info(self) -> Dict[str, Any]:
        """Get information about both LangChain memory and custom conversation buffer."""
        langchain_messages = len(self.langchain_memory.chat_memory.messages)
        return {
            "custom_message_count": self.conversation_buffer.get_message_count(),
            "custom_max_messages": self.conversation_buffer.max_messages,
            "langchain_message_count": langchain_messages,
            "langchain_max_tokens": self.langchain_memory.max_token_limit,
            "recent_context": self.conversation_buffer.get_context_string(max_chars=500)
        }

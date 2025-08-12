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
try:
    from .langgraph_conversation import LangGraphConversation
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LangGraphConversation = None
    LANGGRAPH_AVAILABLE = False


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
        # Include profile query so consistent profile information can be retrieved
        self.context_builder = ContextBuilder(
            memory_engine, profile_query="Jeremy wife Ashley kids dogs age"
        )
        self.logger = get_logger("openai_integration")

        # Initialize LangGraph conversation system (replaces deprecated ConversationChain)
        if LANGGRAPH_AVAILABLE:
            self.langgraph_conversation = LangGraphConversation(
                api_key=api_key,
                memory_engine=memory_engine,
                model=model,
                temperature=0.7
            )
        else:
            self.langgraph_conversation = None
            self.logger.warning("LangGraph not available, falling back to legacy ConversationChain")
        
        # Keep legacy systems for backward compatibility (deprecated)
        self.langchain_chat = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model,
            temperature=0.7
        )
        
        self.langchain_memory = ConversationSummaryBufferMemory(
            llm=self.langchain_chat,
            max_token_limit=8000,  # Increased significantly to preserve immediate context
            return_messages=False,  # Return as string for ConversationChain
            memory_key="history"
        )
        
        # Create custom prompt template that includes our system prompt
        from langchain.prompts import PromptTemplate
        
        # This template combines our system prompt with conversation history
        custom_template = """You're Jeremy's AI assistant with persistent memory. Pay close attention to conversation flow and context.

About Jeremy: 41 years old, wife Ashley, 7 kids, dogs Remy & Bailey. Direct communicator who dislikes generic responses.

CRITICAL CONTEXT RULES:
- Always reference what was JUST discussed in the last few messages
- When Jeremy says "yes"/"okay"/"sure" = confirmation of what was just mentioned
- When Jeremy asks "what do you think"/"which tasks" = refer to specific items just mentioned  
- When Jeremy asks vague questions, connect them to the immediate conversation context
- NEVER give generic advice when specific context exists
- Always check: what tasks, topics, or decisions were mentioned in recent messages?

Current conversation:
{history}
Human: {input}
AI Assistant:"""

        custom_prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=custom_template
        )
        
        # This is the key - use LangChain's ConversationChain with our custom prompt (DEPRECATED)
        self.conversation_chain = ConversationChain(
            llm=self.langchain_chat,
            memory=self.langchain_memory,
            prompt=custom_prompt,  # Use our custom prompt instead of default
            verbose=True  # Enable to debug what LangChain is actually doing
        )

        # Keep custom buffer for compatibility (for now)
        self.conversation_buffer = ConversationBuffer(max_messages=conversation_buffer_size)

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
        thread_id: str = "default",
        use_langgraph: bool = True,
    ) -> str:
        # Use the new LangGraph system by default (if available)
        if use_langgraph and LANGGRAPH_AVAILABLE and self.langgraph_conversation:
            self.logger.info(
                "Starting chat with LangGraph conversation system",
                extra={
                    "message_length": len(message),
                    "thread_id": thread_id,
                    "remember_response": remember_response,
                    "has_system_prompt": system_prompt is not None,
                },
            )

            try:
                # Use the new LangGraph conversation system
                answer = self.langgraph_conversation.chat_with_memory(
                    message=message,
                    thread_id=thread_id,
                    system_prompt=system_prompt,
                    remember_response=remember_response
                )

                self.logger.info(
                    "LangGraph conversation response received",
                    extra={
                        "response_length": len(answer) if answer else 0,
                        "model": self.model,
                        "thread_id": thread_id,
                    },
                )

                return answer

            except Exception as e:
                self.logger.error(
                    "Failed to get LangGraph conversation response",
                    extra={"error": str(e), "model": self.model, "thread_id": thread_id},
                    exc_info=True,
                )
                # Fall back to legacy system
                self.logger.warning("Falling back to legacy ConversationChain system")
                use_langgraph = False
        else:
            # If LangGraph is unavailable, automatically use legacy system
            use_langgraph = False

                # Legacy fallback using direct OpenAI chat completion
        if not use_langgraph:
            self.logger.info(
                "Using legacy LangChain conversation chain",
                extra={
                    "message_length": len(message),
                    "include_recent": include_recent,
                    "include_relevant": include_relevant,
                    "remember_response": remember_response,
                    "has_system_prompt": system_prompt is not None,
                },
            )

            self.logger.debug("Building context from long-term memory")
            context = self.context_builder.build_context(
                query=message,
                include_recent=include_recent,
                include_relevant=include_relevant,
            )

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if context:
                messages.append({"role": "system", "content": f"Previous context: {context}"})
            messages.append({"role": "user", "content": message})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )
            answer = response.choices[0].message.content

            if remember_response:
                self.logger.debug("Storing conversation in persistent memory")
                self.memory_engine.add_memory(f"User: {message}", metadata={"type": "user_message"})
                self.memory_engine.add_memory(f"Assistant: {answer}", metadata={"type": "assistant_response"})

            self.logger.info(
                "Legacy LangChain chat with memory completed successfully",
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

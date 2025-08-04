from typing import List, Dict, Any, Optional
from openai import OpenAI
import numpy as np
from core.memory_engine import Memory, MemoryEngine
from core.context_builder import ContextBuilder
from core.conversation_buffer import ConversationBuffer
from core.logging_config import get_logger, monitor_performance
from .embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
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
        
        self.langchain_memory = ConversationBufferWindowMemory(
            k=conversation_buffer_size//2,  # Number of exchanges (user+AI pairs)
            return_messages=False,  # Return as string for ConversationChain
            memory_key="history"
        )
        
        # This is the key - use LangChain's ConversationChain which handles flow properly
        self.conversation_chain = ConversationChain(
            llm=self.langchain_chat,
            memory=self.langchain_memory,
            verbose=False
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
        """Build a directive system prompt that forces conversation flow understanding"""
        prompt = f"""You're Jeremy's direct AI assistant. CRITICAL: Look at the FULL conversation flow in the messages below.

CONVERSATION FLOW RULES:
1. When Jeremy answers a question, acknowledge his specific answer first
2. Build on his answer, don't ignore it or change topics
3. "force it" = his method for handling tasks - discuss THAT specifically
4. When he asks "what were we talking about" - refer back to the ORIGINAL topic

Jeremy: 7 kids, wife Ashley, dogs Remy & Bailey, age 41. Hates generic responses.

{f"Long-term memories: {context}" if context else ""}
{f"Preferences: {user_preferences}" if user_preferences else ""}

EXAMPLE CONVERSATION FLOW:
Jeremy: "have big tasks to do"
You: "How do you handle them?"  
Jeremy: "force it"
You: "Ah, so you push through by forcing yourself. That's a tough approach - does that strategy work well for you with these big tasks, or does it burn you out?"

NOT: "What's making you feel this way?" (ignoring his answer)"""
        
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
        
        # Build enhanced system prompt and inject it into the conversation chain
        if system_prompt:
            enhanced_system_prompt = system_prompt
        else:
            enhanced_system_prompt = self._build_system_prompt(user_preferences, context)
        
        # Update the conversation chain's prompt template with our enhanced system prompt
        from langchain.prompts import PromptTemplate
        
        custom_prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=f"""{enhanced_system_prompt}

Current conversation:
{{history}}
Human: {{input}}
Assistant:"""
        )
        
        self.conversation_chain.prompt = custom_prompt
        
        self.logger.debug(
            "Using LangChain ConversationChain with enhanced prompt", 
            extra={
                "has_preferences": bool(user_preferences),
                "context_length": len(context) if context else 0,
                "chain_memory_messages": len(self.langchain_memory.chat_memory.messages)
            }
        )

        try:
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
            "langchain_max_messages": self.langchain_memory.k,
            "recent_context": self.conversation_buffer.get_context_string(max_chars=500)
        }

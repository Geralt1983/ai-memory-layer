"""
LangGraph-based conversation system with proper memory management
Replaces the deprecated ConversationChain approach
"""

import os
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from datetime import datetime
import json

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from core.memory_engine import MemoryEngine
from core.logging_config import get_logger, monitor_performance


class ConversationState(TypedDict):
    """State for managing conversation flow"""
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    user_input: Annotated[str, "Current user input"]
    ai_response: Annotated[str, "Current AI response"]
    context: Annotated[str, "Relevant context from memory"]
    system_prompt: Annotated[Optional[str], "System prompt override"]
    metadata: Annotated[Dict[str, Any], "Additional metadata"]


class LangGraphConversation:
    """
    LangGraph-based conversation system with robust memory management
    """
    
    def __init__(
        self,
        api_key: str,
        memory_engine: MemoryEngine,
        model: str = "gpt-4o",
        temperature: float = 0.7
    ):
        self.api_key = api_key
        self.memory_engine = memory_engine
        self.model = model
        self.temperature = temperature
        self.logger = get_logger("langgraph_conversation")
        
        # Initialize OpenAI chat model
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model=model,
            temperature=temperature
        )
        
        # Create memory checkpointer for persistence
        self.checkpointer = MemorySaver()
        
        # Build the conversation graph
        self.graph = self._build_conversation_graph()
        
        self.logger.info(
            "LangGraph conversation system initialized",
            extra={"model": model, "temperature": temperature}
        )
    
    def _build_conversation_graph(self) -> StateGraph:
        """Build the conversation flow graph"""
        
        def retrieve_context(state: ConversationState) -> ConversationState:
            """Retrieve relevant context from memory"""
            user_input = state["user_input"]
            
            try:
                # Get recent context from conversation history with smart prioritization
                all_messages = state["messages"]
                
                # Always keep the most recent messages
                recent_messages = all_messages[-20:] if len(all_messages) > 20 else all_messages
                
                # Also include any messages that contain task/project keywords to maintain context
                important_keywords = ["task", "project", "decision", "status", "approach", "priority", "implement", "finish", "complete"]
                important_messages = []
                
                if len(all_messages) > 20:
                    # Look for important messages in the older history
                    older_messages = all_messages[:-20]
                    for msg in older_messages[-30:]:  # Check last 30 older messages
                        content_lower = msg.content.lower()
                        if any(keyword in content_lower for keyword in important_keywords):
                            important_messages.append(msg)
                
                # Combine important messages with recent ones, avoiding duplicates
                combined_messages = important_messages + recent_messages
                seen_content = set()
                deduplicated_messages = []
                for msg in combined_messages:
                    if msg.content not in seen_content:
                        deduplicated_messages.append(msg)
                        seen_content.add(msg.content)
                
                recent_context = "\n".join([
                    f"{msg.__class__.__name__}: {msg.content}"
                    for msg in deduplicated_messages[-25:]  # Keep max 25 messages total
                ])
                
                # Get relevant memories from persistent storage
                relevant_memories = self.memory_engine.search_memories(user_input, k=5)
                memory_context = "\n".join([
                    f"- {memory.content}" for memory in relevant_memories
                ])
                
                # Combine contexts with better structure
                full_context = ""
                if recent_context:
                    full_context += f"## Recent Context:\n{recent_context}\n\n"
                if memory_context:
                    full_context += f"## Relevant Context:\n{memory_context}"
                
                # Add explicit instruction to use the context
                if full_context:
                    full_context = f"""IMPORTANT: Use the context below to understand what has been discussed. Pay special attention to:
- Tasks, projects, or work Jeremy mentioned
- Decisions or approaches discussed
- Any ongoing topics or priorities

{full_context}"""
                
                state["context"] = full_context
                
                self.logger.debug(
                    "Context retrieved",
                    extra={
                        "recent_messages_count": len(recent_messages),
                        "relevant_memories_count": len(relevant_memories),
                        "context_length": len(full_context)
                    }
                )
                
            except Exception as e:
                self.logger.error("Failed to retrieve context", extra={"error": str(e)})
                state["context"] = ""
            
            return state
        
        def generate_response(state: ConversationState) -> ConversationState:
            """Generate AI response using LLM"""
            user_input = state["user_input"]
            context = state.get("context", "")
            system_prompt = state.get("system_prompt")
            
            try:
                # Build system prompt
                if not system_prompt:
                    system_prompt = self._build_system_prompt(context)
                
                # Prepare messages for LLM
                messages = []
                
                # Add system message
                if system_prompt:
                    messages.append(("system", system_prompt))
                
                # Add recent conversation history (last 12 messages for better context retention)
                recent_messages = state["messages"][-12:]
                for msg in recent_messages:
                    if isinstance(msg, HumanMessage):
                        messages.append(("human", msg.content))
                    elif isinstance(msg, AIMessage):
                        messages.append(("assistant", msg.content))
                
                # Add current user input
                messages.append(("human", user_input))
                
                # Generate response
                response = self.llm.invoke(messages)
                ai_response = response.content
                
                state["ai_response"] = ai_response
                
                self.logger.info(
                    "Response generated",
                    extra={
                        "user_input_length": len(user_input),
                        "response_length": len(ai_response),
                        "context_used": bool(context),
                        "system_prompt_used": bool(system_prompt)
                    }
                )
                
            except Exception as e:
                self.logger.error("Failed to generate response", extra={"error": str(e)})
                state["ai_response"] = "I apologize, but I encountered an error. Please try again."
            
            return state
        
        def update_conversation(state: ConversationState) -> ConversationState:
            """Update conversation history with new messages"""
            user_input = state["user_input"]
            ai_response = state["ai_response"]
            
            # Add new messages to conversation history
            new_messages = state["messages"].copy()
            new_messages.append(HumanMessage(content=user_input))
            new_messages.append(AIMessage(content=ai_response))
            
            state["messages"] = new_messages
            
            self.logger.debug(
                "Conversation updated",
                extra={"total_messages": len(new_messages)}
            )
            
            return state
        
        def store_memories(state: ConversationState) -> ConversationState:
            """Store conversation in persistent memory"""
            user_input = state["user_input"]
            ai_response = state["ai_response"]
            
            try:
                # Store user message
                self.memory_engine.add_memory(
                    f"User: {user_input}",
                    metadata={"type": "user_message", "timestamp": datetime.now().isoformat()}
                )
                
                # Store assistant response
                self.memory_engine.add_memory(
                    f"Assistant: {ai_response}",
                    metadata={"type": "assistant_response", "timestamp": datetime.now().isoformat()}
                )
                
                self.logger.debug("Memories stored successfully")
                
            except Exception as e:
                self.logger.error("Failed to store memories", extra={"error": str(e)})
            
            return state
        
        # Build the graph
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("retrieve_context", retrieve_context)
        workflow.add_node("generate_response", generate_response)
        workflow.add_node("update_conversation", update_conversation)
        workflow.add_node("store_memories", store_memories)
        
        # Define the flow
        workflow.add_edge(START, "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "update_conversation")
        workflow.add_edge("update_conversation", "store_memories")
        workflow.add_edge("store_memories", END)
        
        # Compile with checkpointer for persistence
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _build_system_prompt(self, context: str) -> str:
        """Build system prompt with context"""
        base_prompt = """You're Jeremy's AI assistant with persistent memory. Pay close attention to conversation flow and context throughout the ENTIRE conversation.

About Jeremy: 41 years old, wife Ashley, 7 kids, dogs Remy & Bailey. Direct communicator who dislikes generic responses.

CRITICAL CONTEXT RULES:
- ALWAYS reference what was discussed throughout the conversation, not just recent messages
- When Jeremy says "yes"/"okay"/"sure" = confirmation of what was just mentioned
- When Jeremy asks "what do you think"/"which tasks"/"suggested approach" = refer to specific items mentioned earlier in conversation
- When Jeremy asks vague questions, connect them to ALL relevant conversation context, including earlier topics
- NEVER give generic advice when specific context exists from ANY point in the conversation
- Always check: what tasks, topics, decisions, or approaches were mentioned throughout this conversation?
- Maintain awareness of ongoing projects and tasks mentioned earlier, even if many messages have passed
- If Jeremy references something vaguely, look for it in the entire conversation history provided"""
        
        if context:
            return f"{base_prompt}\n\nCONTEXT:\n{context}"
        else:
            return base_prompt
    
    @monitor_performance("chat_with_memory")
    def chat_with_memory(
        self,
        message: str,
        thread_id: str = "default",
        system_prompt: Optional[str] = None,
        remember_response: bool = True
    ) -> str:
        """
        Chat with the AI using LangGraph memory management
        
        Args:
            message: User's message
            thread_id: Conversation thread ID for persistence
            system_prompt: Optional system prompt override
            remember_response: Whether to store the conversation in memory
            
        Returns:
            AI's response
        """
        try:
            # Prepare initial state
            config = RunnableConfig(configurable={"thread_id": thread_id})
            
            # Get current state to preserve message history
            try:
                current_state = self.graph.get_state(config)
                messages = current_state.values.get("messages", []) if current_state.values else []
            except:
                messages = []
            
            initial_state = {
                "messages": messages,
                "user_input": message,
                "ai_response": "",
                "context": "",
                "system_prompt": system_prompt,
                "metadata": {"remember": remember_response}
            }
            
            # Run the conversation graph
            result = self.graph.invoke(initial_state, config=config)
            
            response = result.get("ai_response", "I apologize, but I couldn't generate a response.")
            
            self.logger.info(
                "Chat completed successfully",
                extra={
                    "thread_id": thread_id,
                    "message_length": len(message),
                    "response_length": len(response),
                    "remember_response": remember_response
                }
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                "Chat failed",
                extra={"error": str(e), "thread_id": thread_id},
                exc_info=True
            )
            return "I apologize, but I encountered an error processing your message. Please try again."
    
    def get_conversation_history(self, thread_id: str = "default") -> List[BaseMessage]:
        """Get conversation history for a thread"""
        try:
            config = RunnableConfig(configurable={"thread_id": thread_id})
            state = self.graph.get_state(config)
            return state.values.get("messages", []) if state.values else []
        except Exception as e:
            self.logger.error("Failed to get conversation history", extra={"error": str(e)})
            return []
    
    def clear_conversation(self, thread_id: str = "default") -> None:
        """Clear conversation history for a thread"""
        try:
            config = RunnableConfig(configurable={"thread_id": thread_id})
            # Reset the state
            self.graph.update_state(
                config,
                {"messages": [], "user_input": "", "ai_response": "", "context": ""}
            )
            self.logger.info("Conversation cleared", extra={"thread_id": thread_id})
        except Exception as e:
            self.logger.error("Failed to clear conversation", extra={"error": str(e)})
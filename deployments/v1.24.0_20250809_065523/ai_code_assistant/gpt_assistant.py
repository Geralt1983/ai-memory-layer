#!/usr/bin/env python3
"""
GPT Assistant
=============

Handles interactions with OpenAI's GPT models for intelligent responses.
Manages conversation context, token limits, and response optimization.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class GPTAssistant:
    """GPT-powered assistant for technical conversations"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", max_tokens: int = 2000):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library required: pip install openai")
        
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.conversation_history: List[Dict[str, str]] = []
        
        # Model configuration
        self.temperature = 0.7  # Balanced creativity vs consistency
        self.top_p = 0.9
        self.frequency_penalty = 0.1  # Reduce repetition
        self.presence_penalty = 0.1   # Encourage diverse responses
        
        logger.info(f"GPT assistant initialized with model {model}")
    
    async def chat(self, prompt: str, include_history: bool = False) -> str:
        """Generate a response using GPT"""
        try:
            # Prepare messages
            messages = []
            
            # Add conversation history if requested
            if include_history and self.conversation_history:
                messages.extend(self.conversation_history[-10:])  # Last 10 exchanges
            
            # Add current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
            
            assistant_response = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            # Keep history manageable (last 20 messages = 10 exchanges)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            logger.info(f"Generated response ({len(assistant_response)} chars, {response.usage.total_tokens} tokens)")
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"GPT API call failed: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    async def chat_streaming(self, prompt: str, include_history: bool = False):
        """Generate streaming response (for real-time UI)"""
        try:
            # Prepare messages
            messages = []
            
            if include_history and self.conversation_history:
                messages.extend(self.conversation_history[-10:])
            
            messages.append({"role": "user", "content": prompt})
            
            # Make streaming API call
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            # Update conversation history with complete response
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": full_response})
            
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
        except Exception as e:
            logger.error(f"Streaming GPT API call failed: {e}")
            yield f"Error: {str(e)}"
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation"""
        if not self.conversation_history:
            return "No conversation history available."
        
        total_exchanges = len(self.conversation_history) // 2
        recent_topics = []
        
        # Extract topics from recent user messages
        for i in range(len(self.conversation_history) - 2, -1, -2):  # Every other message (user messages)
            if len(recent_topics) >= 3:
                break
            
            user_message = self.conversation_history[i]["content"]
            # Extract first sentence or up to 50 characters
            topic = user_message.split('.')[0][:50]
            if len(user_message) > 50:
                topic += "..."
            recent_topics.append(topic)
        
        summary = f"Conversation history: {total_exchanges} exchanges"
        if recent_topics:
            summary += f"\nRecent topics: {', '.join(recent_topics)}"
        
        return summary
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def set_model_parameters(self, **kwargs):
        """Update model parameters"""
        if 'temperature' in kwargs:
            self.temperature = max(0.0, min(2.0, kwargs['temperature']))
        
        if 'max_tokens' in kwargs:
            self.max_tokens = max(1, min(4000, kwargs['max_tokens']))
        
        if 'top_p' in kwargs:
            self.top_p = max(0.0, min(1.0, kwargs['top_p']))
        
        if 'frequency_penalty' in kwargs:
            self.frequency_penalty = max(-2.0, min(2.0, kwargs['frequency_penalty']))
        
        if 'presence_penalty' in kwargs:
            self.presence_penalty = max(-2.0, min(2.0, kwargs['presence_penalty']))
        
        logger.info(f"Updated model parameters: temp={self.temperature}, max_tokens={self.max_tokens}")
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count"""
        # Rough approximation: 1 token ≈ 4 characters for English
        return len(text) // 4
    
    def get_stats(self) -> Dict[str, Any]:
        """Get assistant statistics"""
        total_user_messages = len([m for m in self.conversation_history if m["role"] == "user"])
        total_assistant_messages = len([m for m in self.conversation_history if m["role"] == "assistant"])
        
        total_chars = sum(len(m["content"]) for m in self.conversation_history)
        estimated_tokens = self.estimate_tokens(str(total_chars))
        
        return {
            "model": self.model,
            "conversation_exchanges": total_user_messages,
            "total_messages": len(self.conversation_history),
            "total_characters": total_chars,
            "estimated_tokens_used": estimated_tokens,
            "current_temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "conversation_summary": self.get_conversation_summary()
        }

class MockGPTAssistant:
    """Mock GPT assistant for testing without OpenAI API"""
    
    def __init__(self, model: str = "mock-gpt-4"):
        self.model = model
        self.conversation_history = []
        logger.info("Mock GPT assistant initialized")
    
    async def chat(self, prompt: str, include_history: bool = False) -> str:
        """Generate mock response"""
        # Simple rule-based responses for testing
        prompt_lower = prompt.lower()
        
        if "commit" in prompt_lower:
            response = """Based on the commit information provided, I can see the changes involve code improvements and new features. Here are my observations:

1. **Code Quality**: The changes appear to follow good coding practices
2. **Architecture**: The modifications align with the existing system design
3. **Testing**: Consider adding tests for the new functionality
4. **Documentation**: Update relevant documentation to reflect these changes

Would you like me to elaborate on any specific aspect of these changes?"""
        
        elif "architecture" in prompt_lower or "design" in prompt_lower:
            response = """The AI Memory Layer follows a well-structured architecture with these key components:

1. **Core Memory Engine**: Handles memory storage and retrieval using FAISS
2. **Embedding Service**: Manages text embeddings via OpenAI API
3. **Vector Store**: Efficient storage and similarity search
4. **API Layer**: FastAPI-based REST interface
5. **Webhook Integration**: Automatic processing of GitHub events

This modular design allows for scalability and maintainability. Each component has clear responsibilities and interfaces."""
        
        elif "test" in prompt_lower:
            response = """For testing this system, I recommend:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Verify component interactions
3. **API Tests**: Validate REST endpoints and responses
4. **Performance Tests**: Ensure acceptable response times
5. **Memory Tests**: Verify embedding and retrieval accuracy

Consider using pytest for the test framework and include both positive and negative test cases."""
        
        elif "performance" in prompt_lower:
            response = """To optimize performance in the AI Memory Layer:

1. **FAISS Optimization**: Use appropriate index types (IVF, HNSW)
2. **Caching**: Implement embedding caching to reduce API calls
3. **Batch Processing**: Process multiple items together
4. **Connection Pooling**: Optimize database connections
5. **Async Processing**: Use asyncio for I/O operations

Monitor memory usage and response times to identify bottlenecks."""
        
        else:
            response = f"""I understand you're asking about: "{prompt[:100]}..."

Based on the AI Memory Layer project context, I can help with:
- Code analysis and improvements
- Architecture recommendations  
- Performance optimization
- Testing strategies
- Integration guidance

Could you provide more specific details about what you'd like to explore?"""
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    async def chat_streaming(self, prompt: str, include_history: bool = False):
        """Mock streaming response"""
        response = await self.chat(prompt, include_history)
        # Simulate streaming by yielding chunks
        words = response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
    
    def get_conversation_summary(self) -> str:
        return f"Mock conversation with {len(self.conversation_history) // 2} exchanges"
    
    def clear_conversation_history(self):
        self.conversation_history.clear()
    
    def set_model_parameters(self, **kwargs):
        pass
    
    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "conversation_exchanges": len(self.conversation_history) // 2,
            "total_messages": len(self.conversation_history),
            "total_characters": sum(len(m["content"]) for m in self.conversation_history),
            "estimated_tokens_used": 100,
            "current_temperature": 0.7,
            "max_tokens": 2000,
            "conversation_summary": self.get_conversation_summary()
        }
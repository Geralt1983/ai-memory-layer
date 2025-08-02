from typing import List, Dict, Any, Optional
from openai import OpenAI
import numpy as np
from ..core.memory_engine import Memory, MemoryEngine
from ..core.context_builder import ContextBuilder
from .embeddings import OpenAIEmbeddings


class OpenAIIntegration:
    def __init__(
        self, 
        api_key: str,
        memory_engine: MemoryEngine,
        model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002"
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.memory_engine = memory_engine
        self.embeddings = OpenAIEmbeddings(api_key, embedding_model)
        self.context_builder = ContextBuilder(memory_engine)
    
    def chat_with_memory(
        self, 
        message: str,
        system_prompt: Optional[str] = None,
        include_recent: int = 5,
        include_relevant: int = 5,
        remember_response: bool = True
    ) -> str:
        # Build context from memory
        context = self.context_builder.build_context(
            query=message,
            include_recent=include_recent,
            include_relevant=include_relevant
        )
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if context:
            messages.append({"role": "system", "content": f"Previous context:\n{context}"})
        
        messages.append({"role": "user", "content": message})
        
        # Get response from OpenAI
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        answer = response.choices[0].message.content
        
        # Store the interaction in memory
        if remember_response:
            # Store user message (MemoryEngine will handle embedding generation)
            self.memory_engine.add_memory(
                f"User: {message}",
                metadata={"type": "user_message"}
            )
            
            # Store assistant response (MemoryEngine will handle embedding generation)
            self.memory_engine.add_memory(
                f"Assistant: {answer}",
                metadata={"type": "assistant_response"}
            )
        
        return answer
    
    def add_memory_with_embedding(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Memory:
        # MemoryEngine now handles embedding generation internally
        return self.memory_engine.add_memory(content, metadata)
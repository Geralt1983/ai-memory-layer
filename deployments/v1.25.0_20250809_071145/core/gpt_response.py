"""
GPT-4 Response Generation with Memory Context
"""
import os
from typing import List
import openai
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_gpt_response(user_input: str, memories: List, model: str = "gpt-4-turbo-preview") -> str:
    """
    Generate intelligent response using GPT-4 with memory context
    
    Args:
        user_input: User's query/message
        memories: List of relevant Memory objects from search
        model: OpenAI model to use
    
    Returns:
        AI-generated response incorporating memory context
    """
    try:
        # Extract and format memory context
        context_pieces = []
        for i, memory in enumerate(memories[:5], 1):  # Use top 5 memories
            if len(memory.content) > 20:  # Skip tiny fragments
                context_pieces.append(f"[Memory {i}]: {memory.content[:500]}")
        
        context = "\n\n".join(context_pieces) if context_pieces else "No relevant past conversations found."
        
        # Create system prompt with memory context
        system_prompt = f"""You are an AI assistant with access to the user's past ChatGPT conversations.
        
Based on these relevant past conversations:
{context}

Provide a helpful, contextual response that references this history when relevant.
If the memories don't contain relevant information, still be helpful but mention you don't have specific past context for this query."""

        # Generate response with GPT-4
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Fallback to simple context-based response if GPT-4 fails
        print(f"GPT-4 generation failed: {e}")
        if memories and len(memories) > 0:
            return f"Based on your past conversations, here's relevant context: {memories[0].content[:300]}..."
        else:
            return "I couldn't generate a response with GPT-4, and no relevant memories were found for your query."

def generate_conversation_title(messages: List[dict]) -> str:
    """
    Generate a concise title for a conversation
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
    
    Returns:
        Short descriptive title
    """
    try:
        # Extract key content from messages
        conversation_text = "\n".join([
            f"{msg.get('sender', 'user')}: {msg.get('content', '')[:100]}" 
            for msg in messages[:3]
        ])
        
        if not conversation_text.strip():
            return "New Chat"
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate a 3-5 word title for this conversation:"},
                {"role": "user", "content": conversation_text}
            ],
            temperature=0.5,
            max_tokens=10
        )
        
        title = response.choices[0].message.content.strip()
        return title[:30] if title else "New Chat"
        
    except Exception as e:
        print(f"Title generation failed: {e}")
        # Fallback to first user message
        for msg in messages:
            if msg.get('sender') == 'user':
                content = msg.get('content', '')
                return content[:25] + "..." if len(content) > 25 else content
        return "New Chat"
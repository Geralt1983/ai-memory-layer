"""
Conversation buffer for maintaining recent chat context.

This provides short-term memory (recent conversation) to complement
the long-term vector memory system.
"""

from typing import List, Dict, Optional
from datetime import datetime
import json


class ConversationBuffer:
    """Manages recent conversation messages for context."""
    
    def __init__(self, max_messages: int = 20):
        """
        Initialize conversation buffer.
        
        Args:
            max_messages: Maximum number of messages to keep in buffer
        """
        self.messages: List[Dict[str, str]] = []
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None) -> None:
        """
        Add a message to the conversation buffer.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            timestamp: Optional timestamp, defaults to now
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp.isoformat()
        }
        
        self.messages.append(message)
        
        # Remove oldest messages if we exceed max_messages
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
    
    def get_messages(self, include_system: bool = False) -> List[Dict[str, str]]:
        """
        Get recent messages for OpenAI API format.
        
        Args:
            include_system: Whether to include system messages
            
        Returns:
            List of messages in OpenAI format
        """
        messages = []
        
        for msg in self.messages:
            if msg["role"] in ["user", "assistant"] or include_system:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        return messages
    
    def get_context_string(self, max_chars: int = 2000) -> str:
        """
        Get conversation as a formatted string for context.
        
        Args:
            max_chars: Maximum characters to return
            
        Returns:
            Formatted conversation string
        """
        if not self.messages:
            return ""
        
        context_parts = []
        total_chars = 0
        
        # Build context from most recent messages, working backwards
        for msg in reversed(self.messages):
            role = "User" if msg["role"] == "user" else "Assistant"
            line = f"{role}: {msg['content']}"
            
            if total_chars + len(line) > max_chars and context_parts:
                break
            
            context_parts.insert(0, line)
            total_chars += len(line)
        
        return "\n".join(context_parts)
    
    def clear(self) -> None:
        """Clear all messages from buffer."""
        self.messages.clear()
    
    def get_message_count(self) -> int:
        """Get current number of messages in buffer."""
        return len(self.messages)
    
    def to_dict(self) -> Dict:
        """Export buffer to dictionary for serialization."""
        return {
            "messages": self.messages,
            "max_messages": self.max_messages
        }
    
    def from_dict(self, data: Dict) -> None:
        """Load buffer from dictionary."""
        self.messages = data.get("messages", [])
        self.max_messages = data.get("max_messages", 20)
    
    def save_to_file(self, filepath: str) -> None:
        """Save buffer to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def load_from_file(self, filepath: str) -> None:
        """Load buffer from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.from_dict(data)
        except FileNotFoundError:
            # File doesn't exist yet, start with empty buffer
            pass
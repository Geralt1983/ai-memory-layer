"""
Conversation Endpoints
Handles conversation title generation and conversation management
"""

from fastapi import APIRouter
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.gpt_response import generate_conversation_title as gpt_generate_title

router = APIRouter()

@router.post("/conversations/generate-title") 
async def generate_title(payload: dict):
    """
    Generate intelligent conversation titles using GPT-4 or fallback methods
    """
    try:
        messages = payload.get("messages", [])
        if not messages:
            return {"title": "New Chat"}
        
        # Try GPT-powered title generation first
        try:
            title = gpt_generate_title(messages)
            return {"title": title}
        except Exception as gpt_error:
            print(f"GPT title generation failed: {gpt_error}")
            
        # Fallback: Extract first user message for title
        first_user_msg = None
        for msg in messages:
            if msg.get("sender") == "user":
                first_user_msg = msg.get("content", "")
                break
                
        if first_user_msg:
            # Create a simple title from first message
            title = first_user_msg[:30].strip()
            if len(first_user_msg) > 30:
                title += "..."
            return {"title": title or "New Chat"}
        else:
            return {"title": "New Chat"}
            
    except Exception as e:
        return {"title": "New Chat", "error": str(e)}
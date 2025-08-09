"""
Chat Endpoints
Handles conversational interactions with memory-enhanced responses
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.gpt_response import generate_gpt_response

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    use_gpt: Optional[bool] = True

# Import the memory engine dependency
def get_memory_engine():
    from api.main import memory_engine
    return memory_engine

@router.post("/chat")
async def chat_endpoint(request: ChatRequest, engine=Depends(get_memory_engine)):
    """
    Main chat endpoint with GPT-4 synthesis and relevance-based filtering
    """
    try:
        # Search for relevant memories with higher k for better selection
        results = engine.search_memories(request.message, k=10)
        
        # FIXED: Use relevance scores instead of just length filtering
        quality_memories = []
        for result in results:
            # Get relevance score (higher is better) 
            relevance_score = getattr(result, 'relevance_score', 0.0)
            
            # Use more realistic relevance thresholds (FAISS scores are typically < 1.0)
            if relevance_score > 0.3:  # Good similarity match
                quality_memories.append(result)
            # Also include longer content with decent scores
            elif (len(result.content) > 100 and 
                  relevance_score > 0.2 and
                  not result.content.strip().lower().startswith(("yeah", "okay", "sure", "hmm", "uh", "um"))):
                quality_memories.append(result)
        
        # Option 1: Use GPT-4 for intelligent responses (if available)
        if request.use_gpt and quality_memories:
            try:
                # Generate intelligent response with GPT-4
                response = generate_gpt_response(request.message, quality_memories)
                return {
                    "response": response,
                    "relevant_memories": len(quality_memories),
                    "total_memories": len(engine.memories),
                    "raw_search_results": len(results),
                    "response_type": "gpt-4"
                }
            except Exception as gpt_error:
                print(f"GPT-4 generation failed, falling back: {gpt_error}")
                # Fall through to simple response
        
        # Option 2: Simple context-based response (fallback or if GPT disabled)
        if len(quality_memories) > 0:
            # Extract meaningful context from best memories
            context_pieces = []
            for result in quality_memories[:2]:  # Use top 2 quality memories
                # Take more content for better context
                content = result.content[:400] if len(result.content) > 400 else result.content
                context_pieces.append(f"Previous conversation: {content}")
            
            context = "\\n\\n".join(context_pieces)
            
            # Generate more helpful conversational responses
            if any(word in request.message.lower() for word in ["hello", "hi", "hey"]):
                response = f"Hello! I found relevant context from your {len(engine.memories):,} ChatGPT conversations. Based on your history, you've discussed various topics. How can I help you today?"
                if quality_memories:
                    response += f"\\n\\nRecent relevant context:\\n{context[:300]}..."
            elif "?" in request.message:
                response = f"I found {len(quality_memories)} relevant conversations in your ChatGPT history that might help answer your question.\\n\\n{context[:500]}..."
            else:
                if len(context) > 100:
                    response = f"From your ChatGPT conversation history, here's relevant context:\\n\\n{context[:600]}..."
                else:
                    response = f"I found {len(quality_memories)} related conversations. Here's what might be relevant: {context}"
        else:
            # Fallback when no quality memories found
            response = f"I searched through your {len(engine.memories):,} ChatGPT conversations but didn't find high-quality matches for '{request.message}'. The search found {len(results)} potential matches, but they were too fragmentary to be useful. Try rephrasing your query or asking about specific topics you know you've discussed."
        
        return {
            "response": response,
            "relevant_memories": len(quality_memories),
            "total_memories": len(engine.memories),
            "raw_search_results": len(results),
            "response_type": "context-only"
        }
    except Exception as e:
        return {"error": str(e)}
"""
GPT-4o Optimized Context Builder
Sophisticated memory integration and context management for human-like responses
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from core.memory_engine import Memory


@dataclass
class ContextConfig:
    """Configuration for GPT-4o context building"""
    max_conversation_messages: int = 20
    max_memory_items: int = 5
    memory_relevance_threshold: float = 0.7
    context_window_days: int = 30
    prioritize_recent: bool = True
    include_identity_profile: bool = True
    include_behavior_log: bool = True


class GPT4oContextBuilder:
    """Advanced context builder optimized for GPT-4o's capabilities"""
    
    def __init__(self, config: ContextConfig = None):
        self.config = config or ContextConfig()
        
        # Base identity profile
        self.user_identity = {
            "name": "Jeremy Kimble",
            "role": "IT consultant and AI system builder",
            "preferences": "Direct communication, technical precision, no fluff responses",
            "context": "Building AI memory layer with vector storage and GPT-4o integration",
            "communication_style": "Expects blunt, helpful responses like a capable peer",
            "personal_details": "41 years old, wife Ashley, 7 kids, dogs Remy & Bailey"
        }
        
        # Behavioral expectations for consistency
        self.behavior_expectations = [
            "Avoid caveats and hedge words unless uncertainty is genuine",
            "Provide formatted code when asked, with brief explanations",
            "Speak like a capable peer, not customer support",
            "Reference conversation history naturally without explicit callbacks",
            "Maintain continuity across topics without restarting context",
            "Be concise but complete - no unnecessary elaboration",
            "Handle ambiguity with confident interpretation rather than asking for clarification"
        ]
    
    def create_base_system_prompt(self, custom_prompt: str = None) -> str:
        """Create the foundational system prompt for GPT-4o"""
        if custom_prompt:
            return custom_prompt
            
        return """You are an AI assistant designed to interact like a sharp, experienced, human-like conversation partner. You are modeled after the best traits of GPT-4o, known for memory-aware, emotionally intelligent, and contextually precise responses.

Your goals are:
- Speak naturally, like a fast-thinking, helpful peer
- Remember and subtly incorporate long-term context and preferences
- Avoid repetition, filler phrases, robotic tone, or over-explaining
- Acknowledge what the user implies, not just what they say
- Maintain continuity in tone, voice, and purpose across threads

DO:
- Be concise, clear, and confident
- Use friendly, professional language with optional cleverness
- Handle ambiguity with tact, not hedging
- Reference memory seamlessly and naturally, not by quoting

DON'T:
- Say "As an AI..." or use boilerplate
- Apologize for things that weren't errors
- Repeat user's question before answering
- Over-explain unless asked to

If retrieved memory is provided, treat it as background insight. Use it to inform your response, not dominate it. Only reference it directly if it clarifies the user's intent.

You are currently supporting a user named Jeremy Kimble, an IT consultant who is building a long-memory AI assistant with vector recall using FAISS and OpenAI's GPT-4o API. He values speed, precision, low-fluff responses, and clever utility."""
    
    def create_identity_message(self) -> str:
        """Create identity profile message for personalization"""
        return f"""User Profile: {self.user_identity['name']}, {self.user_identity['role']}. Personal context: {self.user_identity['personal_details']}. Communication style: {self.user_identity['communication_style']}. Current project: {self.user_identity['context']}."""
    
    def create_behavior_message(self) -> str:
        """Create behavior expectations message for consistency"""
        expectations = " | ".join(self.behavior_expectations[:4])  # Keep it concise
        return f"Past interaction patterns: {expectations}"
    
    def process_memories_for_context(self, memories: List[Memory], query: str) -> Optional[str]:
        """Transform raw memories into natural context summaries"""
        if not memories:
            return None
        
        # Filter memories by relevance and recency
        relevant_memories = []
        cutoff_date = datetime.now() - timedelta(days=self.config.context_window_days)
        
        for memory in memories:
            if (memory.relevance_score >= self.config.memory_relevance_threshold and 
                memory.timestamp >= cutoff_date):
                relevant_memories.append(memory)
        
        if not relevant_memories:
            return None
        
        # Group memories by type for natural summarization
        memory_groups = {}
        for memory in relevant_memories[:self.config.max_memory_items]:
            memory_type = memory.metadata.get('type', 'general')
            if memory_type not in memory_groups:
                memory_groups[memory_type] = []
            memory_groups[memory_type].append(memory)
        
        # Create natural summaries
        context_parts = []
        
        for mem_type, mems in memory_groups.items():
            if mem_type == 'preference':
                prefs = [m.content for m in mems[:2]]
                context_parts.append(f"User preferences: {', '.join(prefs)}")
            elif mem_type == 'user_message':
                topics = [self._extract_topic(m.content) for m in mems[:2]]
                context_parts.append(f"Recent discussions: {', '.join(filter(None, topics))}")
            elif mem_type in ['tool', 'technical', 'project']:
                tech_items = [m.content[:60] + "..." if len(m.content) > 60 else m.content for m in mems[:2]]
                context_parts.append(f"Technical context: {', '.join(tech_items)}")
            else:
                general_context = mems[0].content[:80] + "..." if len(mems[0].content) > 80 else mems[0].content
                context_parts.append(f"Context: {general_context}")
        
        return " | ".join(context_parts) if context_parts else None
    
    def _extract_topic(self, content: str) -> str:
        """Extract main topic from message content"""
        # Simple topic extraction - first few meaningful words
        words = content.split()[:6]  
        topic = " ".join(words)
        return topic[:40] + "..." if len(topic) > 40 else topic
    
    def build_conversation_context(self, conversation_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process conversation history for optimal context"""
        if not conversation_history:
            return []
        
        # For GPT-4o, we can be more selective about which messages to include
        processed_history = []
        
        # Always include recent messages
        recent_messages = conversation_history[-self.config.max_conversation_messages:]
        
        if self.config.prioritize_recent:
            # Weight recent messages more heavily
            for i, msg in enumerate(recent_messages):
                # Add slight weighting info for very recent messages
                if i >= len(recent_messages) - 3:  # Last 3 messages
                    processed_history.append(msg)
                else:
                    processed_history.append(msg)
        else:
            processed_history = recent_messages
        
        return processed_history
    
    def build_full_context(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        memories: List[Memory] = None,
        custom_system_prompt: str = None
    ) -> List[Dict[str, str]]:
        """Build complete GPT-4o optimized message context"""
        messages = []
        
        # 1. Primary system prompt
        messages.append({
            "role": "system", 
            "content": self.create_base_system_prompt(custom_system_prompt)
        })
        
        # 2. Identity profile (if enabled)
        if self.config.include_identity_profile:
            messages.append({
                "role": "system", 
                "content": self.create_identity_message()
            })
        
        # 3. Behavior expectations (if enabled)
        if self.config.include_behavior_log:
            messages.append({
                "role": "system", 
                "content": self.create_behavior_message()
            })
        
        # 4. Memory context (sophisticated processing)
        if memories:
            memory_context = self.process_memories_for_context(memories, user_message)
            if memory_context:
                messages.append({
                    "role": "system",
                    "content": f"Background context: {memory_context}"
                })
        
        # 5. Conversation history (processed)
        processed_history = self.build_conversation_context(conversation_history)
        messages.extend(processed_history)
        
        # 6. Current user message
        messages.append({
            "role": "user", 
            "content": user_message
        })
        
        return messages
    
    def get_optimal_api_params(self, message_type: str = "general") -> Dict[str, Any]:
        """Get GPT-4o optimized API parameters based on message type"""
        base_params = {
            "temperature": 0.7,
            "top_p": 1.0,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.25,
            "max_tokens": 1200
        }
        
        # Adjust based on message type
        if message_type == "technical":
            base_params.update({
                "temperature": 0.6,  # More deterministic for technical content
                "presence_penalty": 0.3  # Less creative exploration
            })
        elif message_type == "creative":
            base_params.update({
                "temperature": 0.8,  # More creative
                "presence_penalty": 0.7  # Encourage novel ideas
            })
        elif message_type == "conversational":
            base_params.update({
                "temperature": 0.75,  # Natural conversation feel
                "frequency_penalty": 0.3  # Reduce repetitive responses
            })
        
        return base_params
    
    def analyze_message_type(self, user_message: str) -> str:
        """Analyze message to determine optimal response parameters"""
        message_lower = user_message.lower()
        
        # Technical indicators
        technical_keywords = ["code", "api", "database", "server", "algorithm", "function", "bug", "error", "debug", "implement"]
        if any(keyword in message_lower for keyword in technical_keywords):
            return "technical"
        
        # Creative indicators  
        creative_keywords = ["idea", "creative", "brainstorm", "design", "concept", "innovative", "think of", "come up with"]
        if any(keyword in message_lower for keyword in creative_keywords):
            return "creative"
        
        # Default to conversational
        return "conversational"
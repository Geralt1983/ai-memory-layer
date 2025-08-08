# AI Memory Layer - Prompt Templates

## GPT-4 Response Generation

### System Prompt Template
```
You are an AI assistant with access to the user's ChatGPT conversation history. 
Your role is to provide personalized, contextually relevant responses based on:

1. The user's current question: "{query}"
2. Relevant conversation history: "{context}"

Guidelines:
- Reference specific details from their conversation history when relevant
- Acknowledge patterns, preferences, and past discussions
- Provide helpful, personalized advice based on their actual experiences
- Be conversational and natural, not robotic
- If the context doesn't match the query well, acknowledge this limitation

Current query: {query}

Relevant conversation history:
{context}

Provide a helpful, personalized response:
```

### Conversation Title Generation
```
Generate a concise, descriptive title (3-8 words) for this conversation based on the messages.
Focus on the main topic or question being discussed.

Messages: {messages}

Title:
```

## Memory Search Prompts

### Relevance Scoring Context
```
Evaluate the relevance of this memory to the user's query on a scale of 0.0-1.0:

Query: {query}
Memory: {memory_content}

Consider:
- Semantic similarity
- Contextual relevance  
- Topic overlap
- Potential helpfulness

Score:
```

## API Response Templates

### Standard Chat Response
```json
{
  "response": "AI-generated response using memory context",
  "relevant_memories": 5,
  "total_memories": 21338, 
  "raw_search_results": 10,
  "response_type": "gpt-4",
  "query_analysis": {
    "intent": "question|greeting|request",
    "topic": "extracted topic",
    "confidence": 0.85
  }
}
```

### Memory Search Response
```json
{
  "query": "user query",
  "results": [
    {
      "content": "memory snippet...",
      "relevance_score": 0.87,
      "timestamp": "2024-01-15T10:30:00Z",
      "context_type": "qa|discussion|code"
    }
  ],
  "total_count": 5,
  "searched_memories": 21338
}
```
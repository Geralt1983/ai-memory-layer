#!/usr/bin/env python3
"""
Debug script to test LangChain conversation flow locally
"""

import os
from dotenv import load_dotenv
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

load_dotenv()

def test_conversation_flow():
    """Test the exact same setup we're using in production"""
    
    # Same setup as our integration
    chat = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o",
        temperature=0.7
    )
    
    memory = ConversationSummaryBufferMemory(
        llm=chat,
        max_token_limit=2000,
        return_messages=False,
        memory_key="history"
    )
    
    conversation = ConversationChain(
        llm=chat,
        memory=memory,
        verbose=True
    )
    
    print("=== Testing Conversation Flow ===")
    print("Scenario: User says tired, wants to push through. AI suggests breaks. User asks 'like how?'")
    print()
    
    # Simulate the exact scenario that's failing
    print("1. User: i'm feeling tired, but going to push through.")
    response1 = conversation.predict(input="i'm feeling tired, but going to push through.")
    print(f"   AI: {response1}")
    print()
    
    print("2. User: like how")  
    print(f"   Memory buffer before call: {memory.buffer}")
    print()
    
    response2 = conversation.predict(input="like how")
    print(f"   AI: {response2}")
    print()
    
    print("=== Memory Analysis ===")
    print(f"Messages in memory: {len(memory.chat_memory.messages)}")
    print(f"Buffer content: {memory.buffer}")
    
if __name__ == "__main__":
    test_conversation_flow()
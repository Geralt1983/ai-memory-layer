#!/usr/bin/env python3
"""
Debug what's actually in the LangChain conversation memory
"""

import requests
import json

def test_memory_content():
    """Test what the conversation buffer contains"""
    
    # Test the conversation buffer info endpoint
    response = requests.get("http://18.224.179.36/health")
    print("=== Health Check ===")
    print(json.dumps(response.json(), indent=2))
    
    # Send a test message and check memory
    print("\n=== Testing Conversation Flow ===")
    
    # Message 1: Setup the options
    msg1 = {
        "message": "I have two options for you: A) eat pizza or B) eat salad. Which do you prefer?"
    }
    
    response1 = requests.post("http://18.224.179.36/chat", json=msg1)
    print(f"Response 1: {response1.json()['response']}")
    
    # Message 2: Reference the latter option
    msg2 = {
        "message": "the latter"
    }
    
    response2 = requests.post("http://18.224.179.36/chat", json=msg2)
    print(f"Response 2: {response2.json()['response']}")
    
    # Check if the AI understood "the latter" refers to option B (salad)

if __name__ == "__main__":
    test_memory_content()
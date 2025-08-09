#!/usr/bin/env python3
"""
Test OpenAI API Key
Simple script to verify if the API key in .env file is valid
"""

import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

def test_api_key():
    """Test if the OpenAI API key is valid"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ No API key found in .env file")
        return False
    
    # Show masked key for verification
    masked_key = f"{api_key[:8]}...{api_key[-4:]}"
    print(f"🔑 Testing API key: {masked_key}")
    
    # Set up the client
    client = openai.OpenAI(api_key=api_key)
    
    try:
        # Make a minimal API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        print("✅ API key is valid!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except openai.AuthenticationError as e:
        print(f"❌ Authentication failed: {e}")
        print("\nPossible issues:")
        print("1. The API key may have expired")
        print("2. The key may have been revoked")
        print("3. There might be extra whitespace in the .env file")
        print("4. The key format might be incorrect")
        return False
        
    except openai.RateLimitError as e:
        print(f"⚠️  Rate limit hit (but key is valid): {e}")
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return False

def test_embeddings():
    """Test if embeddings work with the API key"""
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input="test message"
        )
        print("\n✅ Embeddings API also works!")
        print(f"Embedding dimension: {len(response.data[0].embedding)}")
        return True
        
    except Exception as e:
        print(f"\n❌ Embeddings failed: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    print("Testing OpenAI API Key from .env file...\n")
    
    if test_api_key():
        test_embeddings()
    else:
        print("\n💡 To fix this:")
        print("1. Go to https://platform.openai.com/api-keys")
        print("2. Create a new API key")
        print("3. Update the OPENAI_API_KEY in your .env file")
        print("4. Make sure there are no extra spaces or quotes around the key")
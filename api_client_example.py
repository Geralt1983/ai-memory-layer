#!/usr/bin/env python3
"""
Example client for testing the AI Memory Layer API
"""
import requests
import json
import time
from typing import Dict, Any, List, Optional


class MemoryAPIClient:
    """Simple client for the AI Memory Layer API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def create_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new memory"""
        payload = {
            "content": content,
            "metadata": metadata or {}
        }
        response = self.session.post(f"{self.base_url}/memories", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_recent_memories(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories"""
        response = self.session.get(f"{self.base_url}/memories", params={"n": n})
        response.raise_for_status()
        return response.json()
    
    def search_memories(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search memories"""
        payload = {
            "query": query,
            "k": k
        }
        response = self.session.post(f"{self.base_url}/memories/search", json=payload)
        response.raise_for_status()
        return response.json()
    
    def chat(self, message: str, system_prompt: Optional[str] = None, 
             include_recent: int = 5, include_relevant: int = 5,
             remember_response: bool = True) -> Dict[str, Any]:
        """Chat with AI using memory context"""
        payload = {
            "message": message,
            "system_prompt": system_prompt,
            "include_recent": include_recent,
            "include_relevant": include_relevant,
            "remember_response": remember_response
        }
        response = self.session.post(f"{self.base_url}/chat", json=payload)
        response.raise_for_status()
        return response.json()
    
    def clear_memories(self) -> None:
        """Clear all memories"""
        response = self.session.delete(f"{self.base_url}/memories")
        response.raise_for_status()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        response = self.session.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the API client"""
    print("AI Memory Layer API Client Example")
    print("=" * 40)
    
    client = MemoryAPIClient()
    
    try:
        # Health check
        print("1. Checking API health...")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Memory count: {health['memory_count']}")
        print(f"   Vector store: {health.get('vector_store_type', 'Unknown')}")
        
        # Add some memories
        print("\n2. Adding memories...")
        memories_to_add = [
            ("I love Python programming", {"type": "preference", "category": "programming"}),
            ("The weather is nice today", {"type": "observation", "category": "weather"}),
            ("FastAPI is great for building APIs", {"type": "fact", "category": "programming"}),
            ("I need to buy groceries later", {"type": "todo", "category": "personal"})
        ]
        
        for content, metadata in memories_to_add:
            memory = client.create_memory(content, metadata)
            print(f"   Created: {memory['content'][:50]}...")
        
        # Get recent memories
        print("\n3. Getting recent memories...")
        recent = client.get_recent_memories(n=3)
        for i, memory in enumerate(recent, 1):
            print(f"   {i}. {memory['content']}")
            print(f"      Type: {memory['metadata'].get('type', 'unknown')}")
        
        # Search memories
        print("\n4. Searching memories...")
        search_results = client.search_memories("programming", k=2)
        print(f"   Found {search_results['total_count']} results:")
        for memory in search_results['memories']:
            print(f"   - {memory['content']}")
            print(f"     Relevance: {memory['relevance_score']:.2f}")
        
        # Chat with memory context
        print("\n5. Chatting with AI...")
        chat_response = client.chat(
            "What do you know about my programming preferences?",
            system_prompt="You are a helpful assistant with access to user's memories."
        )
        print(f"   AI Response: {chat_response['response']}")
        
        if chat_response.get('context_used'):
            print(f"   Context used: {len(chat_response['context_used'])} characters")
        
        # Get statistics
        print("\n6. Getting statistics...")
        stats = client.get_stats()
        print(f"   Total memories: {stats['total_memories']}")
        print(f"   Vector store entries: {stats['vector_store_entries']}")
        print(f"   Memory types: {stats['memory_types']}")
        
        print("\n✅ API client example completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server.")
        print("   Make sure the server is running: python run_api.py")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP error: {e}")
        if e.response:
            print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def interactive_mode():
    """Interactive mode for testing API"""
    print("AI Memory Layer API - Interactive Mode")
    print("Commands: chat, memory, search, recent, stats, clear, quit")
    print("=" * 50)
    
    client = MemoryAPIClient()
    
    try:
        # Health check
        health = client.health_check()
        print(f"✅ Connected to API (memories: {health['memory_count']})")
    except:
        print("❌ Could not connect to API server")
        return
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == "quit" or command == "exit":
                break
            elif command == "chat":
                message = input("Message: ")
                response = client.chat(message)
                print(f"AI: {response['response']}")
            elif command == "memory":
                content = input("Memory content: ")
                memory_type = input("Type (optional): ").strip()
                metadata = {"type": memory_type} if memory_type else {}
                client.create_memory(content, metadata)
                print("✅ Memory added")
            elif command == "search":
                query = input("Search query: ")
                results = client.search_memories(query)
                print(f"Found {results['total_count']} results:")
                for mem in results['memories']:
                    print(f"  - {mem['content']}")
            elif command == "recent":
                n = input("Number of memories (default 5): ").strip()
                n = int(n) if n.isdigit() else 5
                memories = client.get_recent_memories(n)
                for i, mem in enumerate(memories, 1):
                    print(f"  {i}. {mem['content']}")
            elif command == "stats":
                stats = client.get_stats()
                print(f"Total: {stats['total_memories']}")
                print(f"Types: {stats['memory_types']}")
            elif command == "clear":
                confirm = input("Clear all memories? (y/N): ")
                if confirm.lower() == 'y':
                    client.clear_memories()
                    print("✅ Memories cleared")
            else:
                print("Unknown command. Try: chat, memory, search, recent, stats, clear, quit")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye! 👋")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        main()
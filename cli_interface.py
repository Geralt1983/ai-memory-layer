#!/usr/bin/env python3
"""
AI Memory Layer - Command Line Interface
Simple CLI for interacting with your AI Memory Layer
"""

import os
import sys
import json
import requests
from typing import Optional, Dict, Any
from datetime import datetime
import argparse


class MemoryLayerCLI:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
        
    def check_connection(self) -> bool:
        """Check if API is accessible"""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a new memory"""
        try:
            response = self.session.post(
                f"{self.api_url}/memories",
                json={"content": content, "metadata": metadata or {}}
            )
            if response.status_code == 201:
                print(f"✅ Memory added: {content[:50]}...")
                return True
            else:
                print(f"❌ Failed to add memory: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def search_memories(self, query: str, limit: int = 5) -> None:
        """Search memories"""
        try:
            response = self.session.post(
                f"{self.api_url}/memories/search",
                json={"query": query, "k": limit}
            )
            
            if response.status_code == 200:
                data = response.json()
                memories = data.get("memories", [])
                
                print(f"\n🔍 Search results for '{query}' ({len(memories)} found):")
                print("=" * 60)
                
                if not memories:
                    print("No memories found.")
                    return
                
                for i, memory in enumerate(memories, 1):
                    score = memory.get("relevance_score", 0) * 100
                    timestamp = memory.get("timestamp", "")
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            time_str = dt.strftime("%Y-%m-%d %H:%M")
                        except:
                            time_str = timestamp
                    else:
                        time_str = "Unknown"
                    
                    print(f"\n{i}. [{score:.1f}% match] {memory['content']}")
                    print(f"   📅 {time_str}")
                    
                    if memory.get("metadata"):
                        metadata_str = ", ".join([f"{k}:{v}" for k, v in memory["metadata"].items()])
                        print(f"   🏷️  {metadata_str}")
            else:
                print(f"❌ Search failed: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def chat(self, message: str) -> None:
        """Chat with AI using memory context"""
        try:
            response = self.session.post(
                f"{self.api_url}/chat",
                json={
                    "message": message,
                    "include_recent": 5,
                    "include_relevant": 3,
                    "remember_response": True
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n💬 AI Response:")
                print("=" * 50)
                print(data["response"])
                
                if data.get("context_used"):
                    print(f"\n📚 Memory context was used in this response")
            else:
                print(f"❌ Chat failed: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def get_stats(self) -> None:
        """Show memory statistics"""
        try:
            response = self.session.get(f"{self.api_url}/memories/stats")
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n📊 Memory Statistics:")
                print("=" * 30)
                print(f"Total memories: {data['total_memories']}")
                print(f"Average length: {data['average_content_length']:.1f} chars")
                print(f"Total content: {data['total_content_length']} chars")
                
                if data.get('oldest_memory'):
                    try:
                        oldest = datetime.fromisoformat(data['oldest_memory'].replace('Z', '+00:00'))
                        print(f"Oldest memory: {oldest.strftime('%Y-%m-%d %H:%M')}")
                    except:
                        print(f"Oldest memory: {data['oldest_memory']}")
                
                if data.get('memory_types'):
                    print(f"\nMemory types:")
                    for mem_type, count in data['memory_types'].items():
                        print(f"  • {mem_type}: {count}")
            else:
                print(f"❌ Failed to get stats: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def list_recent(self, limit: int = 10) -> None:
        """List recent memories"""
        try:
            response = self.session.get(f"{self.api_url}/memories?n={limit}")
            
            if response.status_code == 200:
                memories = response.json()
                print(f"\n📝 Recent memories ({len(memories)} shown):")
                print("=" * 50)
                
                for i, memory in enumerate(memories, 1):
                    timestamp = memory.get("timestamp", "")
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            time_str = dt.strftime("%Y-%m-%d %H:%M")
                        except:
                            time_str = timestamp
                    else:
                        time_str = "Unknown"
                    
                    print(f"\n{i}. {memory['content']}")
                    print(f"   📅 {time_str}")
                    
                    if memory.get("metadata"):
                        metadata_str = ", ".join([f"{k}:{v}" for k, v in memory["metadata"].items()])
                        print(f"   🏷️  {metadata_str}")
            else:
                print(f"❌ Failed to list memories: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def export_memories(self, format: str = "json") -> None:
        """Export memories to file"""
        try:
            response = self.session.post(
                f"{self.api_url}/memories/export",
                json={"format": format}
            )
            
            if response.status_code == 200:
                filename = f"memories_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Memories exported to {filename}")
            else:
                print(f"❌ Export failed: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID"""
        try:
            response = self.session.delete(f"{self.api_url}/memories/{memory_id}")
            if response.status_code in (200, 204):
                print(f"🗑️ Memory {memory_id} deleted")
                return True
            else:
                print(f"❌ Failed to delete memory: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def update_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update or create a memory by ID"""
        try:
            response = self.session.put(
                f"{self.api_url}/memories/{memory_id}",
                json={"content": content, "metadata": metadata or {}}
            )
            if response.status_code in (200, 201):
                action = "updated" if response.status_code == 200 else "created"
                print(f"✅ Memory {memory_id} {action}")
                return True
            else:
                print(f"❌ Failed to update memory: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def interactive_mode(self) -> None:
        """Start interactive chat mode"""
        print("\n🧠 AI Memory Layer - Interactive Mode")
        print("=" * 40)
        print("Commands:")
        print("  /help     - Show this help")
        print("  /search   - Search memories")
        print("  /stats    - Show statistics")
        print("  /recent   - Show recent memories")
        print("  /export   - Export memories")
        print("  /quit     - Exit")
        print("  Or just type to chat with AI!")
        print()
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == "/quit":
                    print("👋 Goodbye!")
                    break
                elif user_input == "/help":
                    print("\nAvailable commands:")
                    print("  /search   - Search your memories")
                    print("  /stats    - Show memory statistics")
                    print("  /recent   - Show recent memories")
                    print("  /export   - Export memories to file")
                    print("  /quit     - Exit interactive mode")
                    print("  Or type anything else to chat!")
                elif user_input == "/search":
                    query = input("🔍 Search for: ").strip()
                    if query:
                        self.search_memories(query)
                elif user_input == "/stats":
                    self.get_stats()
                elif user_input == "/recent":
                    self.list_recent()
                elif user_input == "/export":
                    self.export_memories()
                else:
                    # Regular chat
                    print("\n🤖 AI:", end=" ")
                    self.chat(user_input)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break


def main():
    parser = argparse.ArgumentParser(description="AI Memory Layer CLI")
    parser.add_argument("--api-url", default="http://localhost:8000", 
                       help="API URL (default: http://localhost:8000)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Add memory command
    add_parser = subparsers.add_parser("add", help="Add a new memory")
    add_parser.add_argument("content", help="Memory content")
    add_parser.add_argument("--metadata", help="JSON metadata")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with AI")
    chat_parser.add_argument("message", help="Message to send")
    
    # Stats command
    subparsers.add_parser("stats", help="Show memory statistics")
    
    # Recent command
    recent_parser = subparsers.add_parser("recent", help="Show recent memories")
    recent_parser.add_argument("--limit", type=int, default=10, help="Number of memories")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export memories")
    export_parser.add_argument("--format", default="json", choices=["json", "csv", "txt"])

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a memory by ID")
    delete_parser.add_argument("id", help="Memory ID")

    # Update command
    update_parser = subparsers.add_parser("update", help="Update or create a memory")
    update_parser.add_argument("id", help="Memory ID")
    update_parser.add_argument("content", help="New memory content")
    update_parser.add_argument("--metadata", help="JSON metadata")
    
    # Interactive command
    subparsers.add_parser("interactive", help="Start interactive mode")
    
    args = parser.parse_args()
    
    cli = MemoryLayerCLI(args.api_url)
    
    # Check connection
    if not cli.check_connection():
        print(f"❌ Cannot connect to API at {args.api_url}")
        print("Make sure the API server is running:")
        print("  python run_api.py")
        sys.exit(1)
    
    print(f"✅ Connected to AI Memory Layer at {args.api_url}")
    
    if args.command == "add":
        metadata = None
        if args.metadata:
            try:
                metadata = json.loads(args.metadata)
            except json.JSONDecodeError:
                print("❌ Invalid JSON metadata")
                sys.exit(1)
        cli.add_memory(args.content, metadata)
    
    elif args.command == "search":
        cli.search_memories(args.query, args.limit)
    
    elif args.command == "chat":
        cli.chat(args.message)
    
    elif args.command == "stats":
        cli.get_stats()
    
    elif args.command == "recent":
        cli.list_recent(args.limit)

    elif args.command == "export":
        cli.export_memories(args.format)

    elif args.command == "delete":
        cli.delete_memory(args.id)

    elif args.command == "update":
        metadata = None
        if args.metadata:
            try:
                metadata = json.loads(args.metadata)
            except json.JSONDecodeError:
                print("❌ Invalid JSON metadata")
                sys.exit(1)
        cli.update_memory(args.id, args.content, metadata)

    elif args.command == "interactive":
        cli.interactive_mode()
    
    else:
        # No command specified, show help and start interactive mode
        parser.print_help()
        print("\nStarting interactive mode...")
        cli.interactive_mode()


if __name__ == "__main__":
    main()

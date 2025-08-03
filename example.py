import os
from dotenv import load_dotenv
from core import MemoryEngine
from storage import FaissVectorStore, ChromaVectorStore
from integrations import OpenAIIntegration
from integrations.embeddings import OpenAIEmbeddings

# Load environment variables
load_dotenv()


def main():
    # Initialize components
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY in your .env file")
        return

    # Initialize embedding provider
    embedding_provider = OpenAIEmbeddings(api_key, model="text-embedding-ada-002")

    # Choose a vector store (Faiss or Chroma)
    # Option 1: Faiss (in-memory or persistent)
    vector_store = FaissVectorStore(dimension=1536, index_path="./data/faiss_index")

    # Option 2: Chroma (in-memory or persistent)
    # vector_store = ChromaVectorStore(collection_name="memories", persist_directory="./data/chroma_db")

    # Initialize memory engine with embedding provider and persistence
    memory_engine = MemoryEngine(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        persist_path="./data/memories.json",
    )

    # Initialize OpenAI integration
    ai = OpenAIIntegration(
        api_key=api_key, memory_engine=memory_engine, model="gpt-3.5-turbo"
    )

    # Example 1: Add some memories
    print("Adding memories...")
    ai.add_memory_with_embedding(
        "The user's favorite programming language is Python",
        metadata={"type": "user_preference"},
    )
    ai.add_memory_with_embedding(
        "The user is working on an AI memory layer project",
        metadata={"type": "project_info"},
    )
    ai.add_memory_with_embedding(
        "The user prefers concise code examples", metadata={"type": "user_preference"}
    )

    # Example 2: Chat with memory
    print("\nChatting with memory context...")

    # First conversation
    response1 = ai.chat_with_memory(
        "What programming language should I use for this project?",
        system_prompt="You are a helpful AI assistant with access to conversation history.",
    )
    print(f"User: What programming language should I use for this project?")
    print(f"AI: {response1}\n")

    # Second conversation (will remember the first)
    response2 = ai.chat_with_memory(
        "Can you give me a code example?",
        system_prompt="You are a helpful AI assistant. Remember user preferences when providing examples.",
    )
    print(f"User: Can you give me a code example?")
    print(f"AI: {response2}\n")

    # Example 3: Search memories
    print("\nSearching memories...")
    memories = memory_engine.search_memories("programming", k=3)
    for i, memory in enumerate(memories):
        print(f"{i+1}. {memory.content} (relevance: {memory.relevance_score:.2f})")

    # Example 4: Get recent memories
    print("\nRecent memories:")
    recent = memory_engine.get_recent_memories(n=5)
    for i, memory in enumerate(recent):
        print(f"{i+1}. {memory.content}")


def interactive_mode():
    """Run an interactive chat session with memory"""
    # Initialize components
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY in your .env file")
        return

    embedding_provider = OpenAIEmbeddings(api_key, model="text-embedding-ada-002")
    vector_store = ChromaVectorStore(persist_directory="./data/chroma_db")
    memory_engine = MemoryEngine(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        persist_path="./data/memories_interactive.json",
    )
    ai = OpenAIIntegration(api_key=api_key, memory_engine=memory_engine)

    print("AI Memory Layer - Interactive Mode")
    print(
        "Type 'exit' to quit, 'clear' to clear memory, 'memories' to list recent memories\n"
    )

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "clear":
            memory_engine.clear_memories()
            print("Memory cleared.\n")
            continue
        elif user_input.lower() == "memories":
            memories = memory_engine.get_recent_memories(n=10)
            print("\nRecent memories:")
            for memory in memories:
                print(f"- {memory.content}")
            print()
            continue

        response = ai.chat_with_memory(user_input)
        print(f"AI: {response}\n")


if __name__ == "__main__":
    # Run the example
    main()

    # Uncomment to run interactive mode
    # interactive_mode()

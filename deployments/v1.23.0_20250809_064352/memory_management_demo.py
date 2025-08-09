#!/usr/bin/env python3
"""
Memory Management Demo Script

Demonstrates cleanup, archival, and export functionality
"""
import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.memory_engine import MemoryEngine, Memory
from core.memory_manager import (
    MemoryManager,
    create_default_memory_manager,
    AgeBasedCleanup,
    SizeBasedCleanup,
    RelevanceBasedCleanup,
    DuplicateCleanup,
)
from storage.faiss_store import FaissVectorStore
from tests.conftest import MockEmbeddingProvider


def create_sample_memories(count: int = 50) -> list:
    """Create sample memories for testing"""
    memories = []
    now = datetime.now()

    # Different types of memories
    memory_types = [
        "user_message",
        "assistant_response",
        "system_note",
        "preference",
        "fact",
    ]

    for i in range(count):
        # Vary the age, relevance, and types
        age_days = i * 2  # 0 to 100 days old
        relevance = max(0.01, 1.0 - (i * 0.02))  # Decreasing relevance
        memory_type = memory_types[i % len(memory_types)]

        memory = Memory(
            content=f"Sample memory {i}: This is a {memory_type} with some content.",
            timestamp=now - timedelta(days=age_days),
            relevance_score=relevance,
            metadata={
                "type": memory_type,
                "index": i,
                "category": "sample" if i % 3 == 0 else "test",
            },
        )
        memories.append(memory)

    # Add some duplicates
    memories.append(
        Memory(
            content="Duplicate content example",
            timestamp=now - timedelta(days=10),
            metadata={"type": "duplicate"},
        )
    )
    memories.append(
        Memory(
            content="Duplicate content example",
            timestamp=now - timedelta(days=15),
            metadata={"type": "duplicate"},
        )
    )

    return memories


def demonstrate_cleanup_strategies():
    """Demonstrate different cleanup strategies"""
    print("🧹 Memory Cleanup Strategies Demo")
    print("=" * 50)

    # Create memory engine with sample data
    engine = MemoryEngine()
    sample_memories = create_sample_memories(20)
    engine.memories = sample_memories

    print(f"Initial memory count: {len(engine.memories)}")

    # Test different strategies
    strategies = [
        ("Age-based (30 days)", AgeBasedCleanup(max_age_days=30)),
        ("Size-based (10 memories)", SizeBasedCleanup(max_memories=10)),
        ("Relevance-based (0.5 min)", RelevanceBasedCleanup(min_relevance=0.5)),
        ("Duplicate removal", DuplicateCleanup()),
    ]

    for name, strategy in strategies:
        print(f"\n{name}:")

        # Count memories that would be cleaned
        cleanup_count = 0
        for i, memory in enumerate(engine.memories):
            context = {"memory_index": i, "total_memories": len(engine.memories)}
            if strategy.should_cleanup(memory, context):
                cleanup_count += 1

        print(f"  Would clean up: {cleanup_count} memories")
        print(f"  Would keep: {len(engine.memories) - cleanup_count} memories")


def demonstrate_memory_manager():
    """Demonstrate full memory manager functionality"""
    print("\n📦 Memory Manager Demo")
    print("=" * 50)

    # Setup
    embedding_provider = MockEmbeddingProvider()
    vector_store = FaissVectorStore(dimension=10)
    engine = MemoryEngine(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        persist_path="./demo_data/memories.json",
    )

    # Add sample memories
    sample_memories = create_sample_memories(30)
    engine.memories = sample_memories

    print(f"Created {len(engine.memories)} sample memories")

    # Create memory manager
    manager = create_default_memory_manager(engine)
    print(f"Memory manager created with {len(manager.cleanup_strategies)} strategies")

    # Show memory stats before cleanup
    stats = engine.get_memory_stats()
    print(f"\nMemory Statistics (before cleanup):")
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Memory types: {stats['memory_types']}")
    print(f"  Average content length: {stats['average_content_length']:.1f}")
    print(f"  Oldest memory: {stats['oldest_memory'].strftime('%Y-%m-%d')}")
    print(f"  Newest memory: {stats['newest_memory'].strftime('%Y-%m-%d')}")

    # Perform cleanup (dry run first)
    print(f"\n🔍 Dry Run Cleanup:")
    dry_stats = manager.cleanup_memories(dry_run=True)

    print(f"  Would clean: {dry_stats.memories_cleaned} memories")
    print(f"  Would archive: {dry_stats.memories_cleaned} memories")
    print(f"  Would keep: {dry_stats.total_memories_after} memories")
    print(f"  Duration: {dry_stats.duration_ms:.1f}ms")

    # Actual cleanup
    print(f"\n🧹 Actual Cleanup:")
    cleanup_stats = manager.auto_cleanup(
        max_memories=15, max_age_days=40, min_relevance=0.3
    )

    print(f"  Cleaned: {cleanup_stats.memories_cleaned} memories")
    print(f"  Archived: {cleanup_stats.memories_archived} memories")
    print(f"  Remaining: {cleanup_stats.total_memories_after} memories")
    print(f"  Duration: {cleanup_stats.duration_ms:.1f}ms")

    # List archives
    archives = manager.archiver.list_archives()
    print(f"\n📚 Archives:")
    for archive in archives:
        print(
            f"  {Path(archive.archive_path).name}: {archive.memory_count} memories, {archive.size_bytes} bytes"
        )

    # Export memories
    print(f"\n📤 Export Demo:")
    export_formats = ["json", "csv", "txt"]

    for format_name in export_formats:
        export_path = f"./demo_data/export_demo.{format_name}"

        # Create demo_data directory if it doesn't exist
        Path("./demo_data").mkdir(exist_ok=True)

        try:
            manager.export_memories(export_path, format=format_name)
            file_size = Path(export_path).stat().st_size
            print(
                f"  Exported to {format_name.upper()}: {export_path} ({file_size} bytes)"
            )
        except Exception as e:
            print(f"  Export to {format_name.upper()} failed: {e}")


def demonstrate_api_usage():
    """Demonstrate API usage for memory management"""
    print("\n🌐 API Usage Examples")
    print("=" * 50)

    print("Memory Management API Endpoints:")
    print("  GET /memories/stats - Get detailed memory statistics")
    print("  POST /memories/cleanup - Clean up memories with criteria")
    print("  GET /archives - List all memory archives")
    print("  POST /memories/export - Export memories to file")
    print("  POST /archives/{name}/restore - Restore from archive")

    print("\nExample API calls:")

    # Cleanup request example
    cleanup_example = {
        "max_memories": 1000,
        "max_age_days": 90,
        "min_relevance": 0.1,
        "archive_before_cleanup": True,
        "dry_run": False,
    }

    print("\n📋 Cleanup Request:")
    print("POST /memories/cleanup")
    print(json.dumps(cleanup_example, indent=2))

    # Export request example
    export_example = {
        "format": "json",
        "filter_type": "user_message",
        "start_date": "2024-01-01T00:00:00",
        "end_date": "2024-12-31T23:59:59",
    }

    print("\n📤 Export Request:")
    print("POST /memories/export")
    print(json.dumps(export_example, indent=2))


def interactive_demo():
    """Interactive demo mode"""
    print("\n🎮 Interactive Memory Management Demo")
    print("=" * 50)

    # Setup
    engine = MemoryEngine()
    sample_memories = create_sample_memories(25)
    engine.memories = sample_memories

    manager = create_default_memory_manager(engine)

    while True:
        print(f"\nCurrent memories: {len(engine.memories)}")
        print("Commands:")
        print("  1. Show memory stats")
        print("  2. Run cleanup (dry run)")
        print("  3. Run actual cleanup")
        print("  4. List archives")
        print("  5. Export memories")
        print("  6. Add sample memories")
        print("  7. Clear all memories")
        print("  q. Quit")

        choice = input("\nEnter command: ").strip().lower()

        if choice == "q":
            break
        elif choice == "1":
            stats = engine.get_memory_stats()
            print(f"\nMemory Statistics:")
            for key, value in stats.items():
                if isinstance(value, datetime):
                    value = value.strftime("%Y-%m-%d %H:%M:%S")
                print(f"  {key}: {value}")

        elif choice == "2":
            dry_stats = manager.cleanup_memories(dry_run=True)
            print(f"\nDry Run Results:")
            print(f"  Would clean: {dry_stats.memories_cleaned}")
            print(f"  Would keep: {dry_stats.total_memories_after}")

        elif choice == "3":
            max_mem = input("Max memories to keep (default 10): ").strip()
            max_mem = int(max_mem) if max_mem.isdigit() else 10

            stats = manager.auto_cleanup(max_memories=max_mem)
            print(f"\nCleanup Results:")
            print(f"  Cleaned: {stats.memories_cleaned}")
            print(f"  Archived: {stats.memories_archived}")
            print(f"  Remaining: {stats.total_memories_after}")

        elif choice == "4":
            archives = manager.archiver.list_archives()
            print(f"\nArchives ({len(archives)}):")
            for i, archive in enumerate(archives):
                print(f"  {i+1}. {Path(archive.archive_path).name}")
                print(
                    f"     Created: {archive.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                print(
                    f"     Memories: {archive.memory_count}, Size: {archive.size_bytes} bytes"
                )

        elif choice == "5":
            format_choice = (
                input("Export format (json/csv/txt, default json): ").strip().lower()
            )
            if format_choice not in ["json", "csv", "txt"]:
                format_choice = "json"

            export_path = f"./demo_data/interactive_export.{format_choice}"
            Path("./demo_data").mkdir(exist_ok=True)

            try:
                manager.export_memories(export_path, format=format_choice)
                file_size = Path(export_path).stat().st_size
                print(f"\nExported to: {export_path} ({file_size} bytes)")
            except Exception as e:
                print(f"\nExport failed: {e}")

        elif choice == "6":
            count = input("Number of memories to add (default 10): ").strip()
            count = int(count) if count.isdigit() else 10

            new_memories = create_sample_memories(count)
            engine.memories.extend(new_memories)
            print(f"\nAdded {count} memories")

        elif choice == "7":
            confirm = input("Clear all memories? (y/N): ").strip().lower()
            if confirm == "y":
                engine.clear_memories()
                print("\nAll memories cleared")

        else:
            print("Invalid command")


def main():
    """Main demo function"""
    print("🧠 AI Memory Layer - Memory Management Demo")
    print("=" * 60)

    # Create demo data directory
    Path("./demo_data").mkdir(exist_ok=True)

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_demo()
    else:
        # Run all demonstrations
        demonstrate_cleanup_strategies()
        demonstrate_memory_manager()
        demonstrate_api_usage()

        print(f"\n✅ Demo completed! Check ./demo_data/ for exported files.")
        print(f"Run 'python {sys.argv[0]} interactive' for interactive mode.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback

        traceback.print_exc()

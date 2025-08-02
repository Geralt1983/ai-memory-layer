#!/usr/bin/env python3
"""
Simple test runner script for the AI Memory Layer project.
This can be used when pytest is not available or for custom test execution.
"""

import sys
import os
import unittest
import importlib.util

def load_test_modules():
    """Load all test modules from the tests directory"""
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    test_modules = []
    
    for filename in os.listdir(test_dir):
        if filename.startswith('test_') and filename.endswith('.py'):
            module_name = filename[:-3]  # Remove .py extension
            module_path = os.path.join(test_dir, filename)
            
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                    test_modules.append(module)
                    print(f"Loaded test module: {module_name}")
                except Exception as e:
                    print(f"Failed to load {module_name}: {e}")
    
    return test_modules

def run_basic_tests():
    """Run basic tests without pytest"""
    print("AI Memory Layer - Basic Test Runner")
    print("=" * 40)
    
    # Add the project root to Python path
    project_root = os.path.dirname(__file__)
    sys.path.insert(0, project_root)
    
    try:
        # Test basic imports
        print("\n1. Testing imports...")
        from core.memory_engine import Memory, MemoryEngine
        from storage.faiss_store import FaissVectorStore
        from integrations.embeddings import EmbeddingProvider
        print("✓ All core imports successful")
        
        # Test Memory class
        print("\n2. Testing Memory class...")
        memory = Memory(content="Test memory", metadata={"type": "test"})
        assert memory.content == "Test memory"
        assert memory.metadata["type"] == "test"
        
        # Test serialization
        data = memory.to_dict()
        restored_memory = Memory.from_dict(data)
        assert restored_memory.content == memory.content
        print("✓ Memory class tests passed")
        
        # Test MemoryEngine
        print("\n3. Testing MemoryEngine...")
        engine = MemoryEngine()
        engine.add_memory("Test content", {"source": "test"})
        assert len(engine.memories) == 1
        assert engine.memories[0].content == "Test content"
        
        recent = engine.get_recent_memories(n=1)
        assert len(recent) == 1
        print("✓ MemoryEngine tests passed")
        
        # Test FAISS store
        print("\n4. Testing FAISS store...")
        import numpy as np
        store = FaissVectorStore(dimension=10)
        test_memory = Memory(
            content="FAISS test",
            embedding=np.random.rand(10).astype('float32')
        )
        memory_id = store.add_memory(test_memory)
        assert store.index.ntotal == 1
        print("✓ FAISS store tests passed")
        
        print("\n" + "=" * 40)
        print("✅ All basic tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed.")
        return False
    except AssertionError as e:
        print(f"❌ Test assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking dependencies...")
    
    required_packages = [
        ('numpy', 'numpy'),
        ('faiss', 'faiss-cpu'),
        ('openai', 'openai'),
        ('pandas', 'pandas'),
        ('sklearn', 'scikit-learn')
    ]
    
    missing = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"❌ {package_name} (missing)")
            missing.append(package_name)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

if __name__ == "__main__":
    print("AI Memory Layer Test Runner")
    print("=" * 50)
    
    if not check_dependencies():
        sys.exit(1)
    
    print("\n" + "=" * 50)
    
    success = run_basic_tests()
    sys.exit(0 if success else 1)
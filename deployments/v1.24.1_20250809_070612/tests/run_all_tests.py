#!/usr/bin/env python3
"""
Test runner for all AI Memory Layer tests
"""

import sys
import subprocess
import os
from pathlib import Path


def run_test_file(test_file: str, verbose: bool = True) -> bool:
    """Run a single test file"""
    cmd = [sys.executable, "-m", "pytest", test_file]
    if verbose:
        cmd.append("-v")
    
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return False


def main():
    """Run all tests"""
    # Set environment variables for testing
    os.environ['SKIP_FAISS_FILE_TEST'] = 'true'  # Skip FAISS file tests if FAISS not available
    os.environ['PYTEST_RUNNING'] = 'true'
    
    test_dir = Path(__file__).parent
    
    # List of test files to run
    test_files = [
        "test_faiss_dimension_guard.py",
        "test_simhash_deduplication.py", 
        "test_embedding_cache.py",
        "test_mmr_diversity.py",
        "test_api_enhancements.py"
    ]
    
    print("AI Memory Layer - Comprehensive Test Suite")
    print("=" * 50)
    
    results = {}
    
    # Run each test file
    for test_file in test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            success = run_test_file(str(test_path))
            results[test_file] = success
        else:
            print(f"Warning: Test file not found: {test_file}")
            results[test_file] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = 0
    total = len(results)
    
    for test_file, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_file}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} test files passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
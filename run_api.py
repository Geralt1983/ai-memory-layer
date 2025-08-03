#!/usr/bin/env python3
"""
Simple script to run the AI Memory Layer API server
"""
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_environment():
    """Check if required environment variables are set"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables in your .env file or environment")
        return False

    return True


def main():
    print("AI Memory Layer API Server")
    print("=" * 40)

    # Load environment variables from .env if available
    try:
        from dotenv import load_dotenv

        if os.path.exists(".env"):
            load_dotenv()
            print("✓ Loaded environment variables from .env")
        else:
            print("! No .env file found, using system environment variables")
    except ImportError:
        print("! python-dotenv not installed, using system environment variables")

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Import and run the API
    try:
        import uvicorn
        from api.main import app

        print("\n🚀 Starting API server...")
        print("📝 API Documentation: http://localhost:8000/docs")
        print("❤️  Health Check: http://localhost:8000/health")
        print("🔍 OpenAPI Schema: http://localhost:8000/openapi.json")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 40)

        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install fastapi uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

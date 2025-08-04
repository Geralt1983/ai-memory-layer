# Changelog

All notable changes to the AI Memory Layer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.2] - 2025-08-04

### Changed
- fix: completely rewrote system prompt with explicit conversation flow example to fix context issues

## [1.3.1] - 2025-08-04

### Changed
- fix: auto-update version number in web interface during deployment

## [1.3.0] - 2025-08-04

### Changed
- feat: upgraded to GPT-4o for much better conversation context and understanding

## [1.2.2] - 2025-08-04

### Changed
- fix: improved system prompt to better follow conversation flow and acknowledge user answers

## [1.2.1] - 2025-08-04

### Changed
- fix: fixed web interface deployment to copy updated HTML to nginx directory

## [1.2.0] - 2025-08-04

### Changed
- feat: added version display (v1.1.2) and reload button to web interface, added debug logging

## [1.1.2] - 2025-08-04

### Changed
- fix: enhanced system prompt and context builder for more specific, helpful AI responses

## [1.1.1] - 2025-08-04

### Changed
- fix: simplified system prompt and removed duplicate conversation context for better AI responses

## [1.1.0] - 2025-08-04

### Changed
- feat: added conversation buffer for short-term memory context

## [1.0.1] - 2025-08-04

### Changed
- EC2 set up. Git deploy and track

## [1.0.0] - 2024-08-04

### Added
- Production deployment on AWS EC2 with permanent availability
- ChatGPT-like web interface with dark mode and conversation management
- Message action buttons (Regenerate, Copy) that appear on hover
- Stop generation button during AI response streaming
- Automatic conversation title generation using GPT-3.5
- systemd service configuration for auto-restart
- nginx reverse proxy for unified web access
- Conversation persistence in browser localStorage
- Enhanced memory context to prevent re-asking known information

### Fixed
- AI hallucination issues (incorrect age and dog names)
- JavaScript error preventing conversation title generation
- Memory contamination from AI's own incorrect responses
- Connection status check in web interface
- FAISS index persistence across restarts

### Changed
- Improved system prompt to better use memory context
- Enhanced OpenAI integration with preference extraction
- Updated web interface to match modern AI chat UIs
- Optimized memory search with relevance scoring

## [0.9.0] - 2024-08-03

### Added
- Core memory engine with persistent storage
- FAISS vector storage implementation
- ChromaDB storage backend option
- OpenAI integration for embeddings (ada-002)
- OpenAI chat completion with memory context
- REST API with FastAPI
- Memory search with semantic similarity
- Memory cleanup strategies (age, size, relevance, duplicates)
- Memory archival and restoration features
- Export functionality (JSON, CSV, TXT)
- CLI interface for command-line usage
- Comprehensive test suite with pytest
- Structured JSON logging with rotation

### Technical Details
- Python 3.8+ support
- Type hints throughout codebase
- Async/await for API endpoints
- Environment-based configuration
- Docker-ready structure

## [0.8.0] - 2024-08-02

### Added
- Initial project structure
- Basic memory storage design
- Requirements specification
- Development environment setup

---

## Versioning Guidelines

- **Major version** (X.0.0): Breaking API changes, major architectural changes
- **Minor version** (0.X.0): New features, backwards-compatible changes
- **Patch version** (0.0.X): Bug fixes, minor improvements

## Upgrade Notes

### From 0.9.0 to 1.0.0
1. Clear all memories to avoid hallucination issues
2. Update web interface to use new enhanced version
3. Deploy using systemd for production use

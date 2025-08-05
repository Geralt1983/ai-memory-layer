# Changelog

All notable changes to the AI Memory Layer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.12.0] - 2025-08-05

### Changed
- feat: integrate ChatGPT memory database with API

## [1.11.7] - 2025-08-04

### Changed
- 

## [1.11.6] - 2025-08-04

### Changed
- 

## [1.11.5] - 2025-08-04

### Added
- feat: memory deduplication system prevents redundant preference recalls
- feat: enhanced identity anchoring with explicit name/style references for thread starts
- feat: varied memory phrasing to eliminate robotic repetition
- feat: balanced identity progression from strong framing to natural conversation flow

### Fixed
- fix: eliminates repeated "Jeremy prefers..." phrasing between responses
- fix: stronger personal identity framing at conversation beginnings
- fix: mid-thread identity balance maintains personal tone without over-anchoring

## [1.11.4] - 2025-08-04

### Changed
- 

## [1.11.3] - 2025-08-04

### Added
- feat: semantic drift prevention with context anchoring for ambiguous follow-ups
- feat: 34+ ambiguous follow-up pattern detection ("what do you think", "sure", "depends", etc.)
- feat: conversation hook system to tie vague responses back to specific topics
- feat: enhanced context anchoring for extra-vague patterns with full thread context

### Fixed
- fix: prevents generic AI responses when users give ambiguous follow-ups
- fix: maintains conversation continuity for short replies like "yeah", "sure", "what do you think"
- fix: context decoding failures where short responses were misinterpreted

## [1.11.2] - 2025-08-04

### Changed
- 

## [1.11.1] - 2025-08-04

### Changed
- 

## [1.11.0] - 2025-08-04

### Added
- feat: GPT-4o optimized system prompts for human-like responses
- feat: sophisticated memory injection with natural context summaries
- feat: identity profile and behavior log integration for consistent personality
- feat: advanced context builder with message type detection
- feat: optimized API parameters (temperature, presence_penalty, frequency_penalty)

### Changed
- Enhanced DirectOpenAIChat with GPT-4o specific optimizations
- Improved conversation continuity with multi-layered system messages
- Updated default model to GPT-4o for superior context handling
- Refined memory integration to feel natural rather than mechanical

## [1.10.0] - 2025-08-04

### Changed
- feat: implement direct OpenAI chat - clean architecture without LangChain/LangGraph

## [1.9.2] - 2025-08-04

### Changed
- fix: persist conversation thread_id across page reloads to maintain context

## [1.9.1] - 2025-08-04

### Changed
- fix: enhanced LangGraph context retention with smart message prioritization

## [1.9.0] - 2025-08-04

### Fixed
- Enhanced LangGraph context retention with smart message prioritization (20→25 messages)
- Improved system prompt to maintain awareness throughout entire conversation history
- Added keyword-based important message preservation for long conversations
- Better context structure with explicit instructions for contextual reference handling

## [1.8.9] - 2025-08-04

### Fixed
- Implemented thread_id support in web interface and API for proper conversation context retention
- Successfully deployed LangGraph-based conversation system to replace deprecated ConversationChain
- Verified contextual reference handling now works correctly ("what do you think", "sure", "which one", etc.)
- Added fallback mechanism for LangGraph dependencies with graceful degradation to legacy system

## [1.8.8] - 2025-08-04

### Changed
- 

## [1.8.7] - 2025-08-04

### Changed
- 

## [1.8.6] - 2025-08-04

### Changed
- 

## [1.8.5] - 2025-08-04

### Changed
- 

## [1.8.4] - 2025-08-04

### Changed
- 

## [1.8.3] - 2025-08-04

### Changed
- 

## [1.8.2] - 2025-08-04

### Changed
- 

## [1.8.1] - 2025-08-04

### Changed
- 

## [1.7.3] - 2025-08-04

### Changed
- 

## [1.7.1] - 2025-08-04

### Changed
- 

## [1.6.1] - 2025-08-04

### Changed
- 

## [1.5.0] - 2025-08-04

### Changed
- feat: replaced raw OpenAI API with LangChain ConversationChain for proper conversation flow handling

## [1.4.0] - 2025-08-04

### Changed
- feat: integrated LangChain ConversationBufferWindowMemory for better conversation context handling

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

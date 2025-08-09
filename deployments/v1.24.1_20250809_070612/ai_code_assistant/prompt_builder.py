#!/usr/bin/env python3
"""
Prompt Builder
==============

Constructs intelligent prompts for the GPT assistant using retrieved memories
and project context. Optimizes for code understanding and technical discussions.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from vector_store import Memory

logger = logging.getLogger(__name__)

class PromptBuilder:
    """Builds optimized prompts for the AI assistant"""
    
    def __init__(self):
        self.max_context_length = 8000  # Characters to include from memories
        self.max_memories = 10
        logger.info("Prompt builder initialized")
    
    def build_prompt(self, query: str, memories: List[Memory], include_system_info: bool = True) -> str:
        """Build a complete prompt with context and query"""
        
        # Start with system message
        prompt_parts = []
        
        if include_system_info:
            prompt_parts.append(self._build_system_message())
        
        # Add memory context
        if memories:
            context_section = self._build_context_section(memories)
            prompt_parts.append(context_section)
        
        # Add current query
        prompt_parts.append(self._build_query_section(query))
        
        # Combine all parts
        full_prompt = "\n\n".join(prompt_parts)
        
        logger.debug(f"Built prompt with {len(memories)} memories, {len(full_prompt)} characters")
        return full_prompt
    
    def _build_system_message(self) -> str:
        """Build the system message that defines the assistant's role"""
        return """You are an expert AI code assistant for the AI Memory Layer project. You have deep knowledge of the entire codebase and its evolution through commit history.

**Your Role:**
- Expert code reviewer and architectural advisor
- Debug assistant with full project context  
- Performance optimization consultant
- Technical documentation expert
- Testing and best practices guide

**Project Overview:**
The AI Memory Layer is a sophisticated Python system that provides:
- FAISS-based vector storage for semantic memory
- OpenAI embeddings and GPT integration
- FastAPI REST API with comprehensive endpoints
- GitHub webhook integration for automatic updates
- Real-time memory indexing and retrieval
- Advanced query processing and context building

**Key Components:**
- Core memory engine with FAISS vector storage
- OpenAI embedding integration (text-embedding-3-small)
- FastAPI web services and endpoints
- GitHub webhook receivers for automatic updates
- Comprehensive testing suite with pytest
- Automated deployment and CI/CD pipeline

**Communication Style:**
- Be direct and technically precise
- Reference specific commits, files, and code patterns
- Provide concrete examples and code snippets
- Ask clarifying questions when needed
- Focus on actionable insights and improvements
- Consider both immediate fixes and long-term architecture"""

    def _build_context_section(self, memories: List[Memory]) -> str:
        """Build the context section from retrieved memories"""
        if not memories:
            return ""
        
        context_parts = ["**Relevant Context from Project Memory:**"]
        
        current_length = len(context_parts[0])
        memories_included = 0
        
        # Group memories by type for better organization
        commit_memories = [m for m in memories if m.metadata.get('type') == 'commit']
        other_memories = [m for m in memories if m.metadata.get('type') != 'commit']
        
        # Prioritize commit memories
        ordered_memories = commit_memories + other_memories
        
        for i, memory in enumerate(ordered_memories):
            if memories_included >= self.max_memories:
                break
            
            memory_section = self._format_memory(memory, i + 1)
            
            # Check if adding this memory would exceed length limit
            if current_length + len(memory_section) > self.max_context_length:
                # Try to include a truncated version
                truncated_content = self._truncate_memory_content(memory, 300)
                if truncated_content:
                    memory_section = f"\n{i + 1}. {self._format_memory_header(memory)}\n{truncated_content}..."
                    if current_length + len(memory_section) <= self.max_context_length:
                        context_parts.append(memory_section)
                        current_length += len(memory_section)
                        memories_included += 1
                break
            
            context_parts.append(memory_section)
            current_length += len(memory_section)
            memories_included += 1
        
        if memories_included < len(memories):
            context_parts.append(f"\n[{len(memories) - memories_included} additional memories available but truncated for brevity]")
        
        return "\n".join(context_parts)
    
    def _format_memory(self, memory: Memory, index: int) -> str:
        """Format a single memory for inclusion in prompt"""
        header = f"\n{index}. {self._format_memory_header(memory)}"
        
        # Format content with appropriate truncation
        content = self._format_memory_content(memory)
        
        return f"{header}\n{content}"
    
    def _format_memory_header(self, memory: Memory) -> str:
        """Format the header for a memory"""
        memory_type = memory.metadata.get('type', 'unknown')
        
        if memory_type == 'commit':
            sha = memory.metadata.get('sha', 'unknown')[:8]
            author = memory.metadata.get('author', 'unknown')
            files_changed = memory.metadata.get('files_changed', [])
            file_count = len(files_changed)
            
            header = f"**Commit {sha}** by {author}"
            if file_count > 0:
                if file_count == 1:
                    header += f" - {files_changed[0]}"
                else:
                    header += f" - {file_count} files changed"
            
            # Add similarity score if available
            if hasattr(memory, 'similarity') and memory.similarity:
                header += f" (relevance: {memory.similarity:.2f})"
            
            return header
        
        elif memory_type == 'conversation':
            return f"**Previous Conversation** (relevance: {getattr(memory, 'similarity', 0.5):.2f})"
        
        else:
            return f"**{memory_type.title()}**"
    
    def _format_memory_content(self, memory: Memory) -> str:
        """Format memory content with smart truncation"""
        content = memory.content.strip()
        
        # For commit memories, focus on the most relevant parts
        if memory.metadata.get('type') == 'commit':
            return self._format_commit_content(content)
        
        # For other memories, use standard formatting
        max_length = 800
        if len(content) <= max_length:
            return content
        
        # Truncate intelligently
        return self._smart_truncate(content, max_length)
    
    def _format_commit_content(self, content: str) -> str:
        """Format commit content, focusing on code changes"""
        lines = content.split('\n')
        
        # Extract key sections
        sections = {
            'summary': [],
            'files': [],
            'code': []
        }
        
        current_section = 'summary'
        code_block_count = 0
        max_code_blocks = 3
        
        for line in lines:
            if line.startswith('## 📁 Changed Files'):
                current_section = 'files'
                sections[current_section].append(line)
            elif line.startswith('### '):
                current_section = 'files'
                sections[current_section].append(line)
            elif line.startswith('```') and current_section == 'files':
                if code_block_count < max_code_blocks:
                    current_section = 'code'
                    sections[current_section].append(line)
                    code_block_count += 1
                else:
                    # Skip additional code blocks
                    continue
            elif current_section == 'code' and line.startswith('```'):
                sections[current_section].append(line)
                current_section = 'files'
            else:
                sections[current_section].append(line)
        
        # Reconstruct content with limits
        result_lines = []
        
        # Add summary (first 5 lines)
        result_lines.extend(sections['summary'][:5])
        
        # Add file changes (up to 10 lines)
        if sections['files']:
            result_lines.extend(sections['files'][:10])
        
        # Add code (already limited by max_code_blocks)
        if sections['code']:
            result_lines.extend(sections['code'])
            if code_block_count >= max_code_blocks:
                result_lines.append("[Additional code blocks truncated for brevity]")
        
        return '\n'.join(result_lines)
    
    def _smart_truncate(self, content: str, max_length: int) -> str:
        """Intelligently truncate content at natural boundaries"""
        if len(content) <= max_length:
            return content
        
        # Try to truncate at paragraph boundaries
        paragraphs = content.split('\n\n')
        result = ""
        
        for paragraph in paragraphs:
            if len(result + paragraph) <= max_length - 20:  # Leave room for ellipsis
                result += paragraph + '\n\n'
            else:
                break
        
        if result:
            return result.rstrip() + "..."
        
        # Fallback: truncate at sentence boundaries
        sentences = content.split('. ')
        result = ""
        
        for sentence in sentences:
            if len(result + sentence) <= max_length - 20:
                result += sentence + '. '
            else:
                break
        
        if result:
            return result.rstrip() + "..."
        
        # Final fallback: hard truncate
        return content[:max_length - 3] + "..."
    
    def _truncate_memory_content(self, memory: Memory, max_length: int) -> str:
        """Get truncated version of memory content"""
        content = memory.content.strip()
        if len(content) <= max_length:
            return content
        
        return self._smart_truncate(content, max_length)
    
    def _build_query_section(self, query: str) -> str:
        """Build the query section of the prompt"""
        return f"""**Current Question:**
{query}

**Instructions:**
Please provide a comprehensive, technical response based on the project context above. Include:
- Direct answers to the specific question
- References to relevant commits or code when applicable  
- Code examples or suggestions where helpful
- Any additional insights or considerations
- Follow-up questions if clarification is needed

Focus on being accurate, actionable, and helpful for continued development of the AI Memory Layer project."""

    def build_simple_prompt(self, query: str) -> str:
        """Build a simple prompt without memory context"""
        return f"""{self._build_system_message()}

**Current Question:**
{query}

**Instructions:**
Please provide a helpful response based on your knowledge of the AI Memory Layer project. Be technical, direct, and actionable in your advice."""

    def get_prompt_stats(self, prompt: str) -> Dict[str, Any]:
        """Get statistics about the generated prompt"""
        return {
            "total_length": len(prompt),
            "word_count": len(prompt.split()),
            "line_count": len(prompt.split('\n')),
            "estimated_tokens": len(prompt) // 4  # Rough estimate
        }
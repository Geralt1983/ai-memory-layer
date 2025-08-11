#!/usr/bin/env python3
"""
Optimized Query Handler with True Vector Search
==============================================

Uses semantic search to find only the most relevant chunks.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from vector_store import Memory
from embedder import EmbeddingService
from memory_query import MemoryQuery
from prompt_builder import PromptBuilder
from gpt_assistant import GPTAssistant

logger = logging.getLogger(__name__)

class OptimizedQueryHandler:
    """Handles queries with optimized vector search"""
    
    def __init__(
        self,
        memory_query: MemoryQuery,
        prompt_builder: PromptBuilder,
        gpt_assistant: GPTAssistant,
        max_context_size: int = 5,  # Only top 5 chunks by default
        enable_reranking: bool = True
    ):
        self.memory_query = memory_query
        self.prompt_builder = prompt_builder
        self.gpt_assistant = gpt_assistant
        self.max_context_size = max_context_size
        self.enable_reranking = enable_reranking
        
    async def handle_query(self, query: str) -> Dict[str, Any]:
        """Process query with optimized retrieval"""
        start_time = time.time()
        
        try:
            # Step 1: Semantic search for top-k relevant chunks
            logger.info(f"Searching for top {self.max_context_size} chunks for query: {query[:50]}...")
            
            # Get more candidates than needed for reranking
            search_size = self.max_context_size * 3 if self.enable_reranking else self.max_context_size
            
            relevant_chunks = await self.memory_query.search(
                query=query,
                max_results=search_size
            )
            
            # Step 2: Apply smart filters based on query intent
            filtered_chunks = self._apply_query_filters(query, relevant_chunks)
            
            # Step 3: Rerank if enabled
            if self.enable_reranking and len(filtered_chunks) > self.max_context_size:
                final_chunks = self._rerank_chunks(query, filtered_chunks)[:self.max_context_size]
            else:
                final_chunks = filtered_chunks[:self.max_context_size]
            
            logger.info(f"Selected {len(final_chunks)} chunks after filtering/reranking")
            
            # Step 4: Build optimized prompt
            prompt = self._build_optimized_prompt(query, final_chunks)
            
            # Step 5: Get GPT response
            response = await self.gpt_assistant.chat(prompt)
            
            # Calculate metrics
            retrieval_time = time.time() - start_time
            
            return {
                "response": response,
                "sources": self._format_sources(final_chunks),
                "metrics": {
                    "retrieval_time": retrieval_time,
                    "chunks_searched": len(relevant_chunks),
                    "chunks_used": len(final_chunks),
                    "avg_similarity": sum(c.similarity or 0 for c in final_chunks) / len(final_chunks) if final_chunks else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Query handling failed: {e}")
            raise
    
    def _apply_query_filters(self, query: str, chunks: List[Memory]) -> List[Memory]:
        """Apply intelligent filters based on query intent"""
        query_lower = query.lower()
        filtered = chunks
        
        # File-specific queries
        file_patterns = {
            'file': r'(?:file|in|from)\s+(\S+\.(?:py|js|ts|md|yaml|json))',
            'extension': r'\.(?:py|js|ts|md|yaml|json)\s+files?'
        }
        
        for pattern_type, pattern in file_patterns.items():
            import re
            match = re.search(pattern, query_lower)
            if match:
                if pattern_type == 'file':
                    target_file = match.group(1)
                    filtered = [c for c in filtered if target_file in c.metadata.get('file_path', '')]
                elif pattern_type == 'extension':
                    ext = match.group(0).split('.')[1].split()[0]
                    filtered = [c for c in filtered if c.metadata.get('file_type') == ext]
        
        # Time-based queries
        time_keywords = {
            'latest': 1,
            'recent': 7,
            'last week': 7,
            'last month': 30,
            'yesterday': 1
        }
        
        for keyword, days in time_keywords.items():
            if keyword in query_lower:
                cutoff = datetime.now() - timedelta(days=days)
                filtered = [c for c in filtered 
                           if self._parse_timestamp(c.metadata.get('date', '')) > cutoff]
                break
        
        # Author-specific queries
        if 'by' in query_lower:
            author_match = re.search(r'by\s+(\w+)', query_lower)
            if author_match:
                author = author_match.group(1)
                filtered = [c for c in filtered 
                           if author.lower() in c.metadata.get('author', '').lower()]
        
        # Chunk type preferences
        if any(word in query_lower for word in ['summary', 'overview', 'description']):
            # Prefer overview and summary chunks
            filtered.sort(key=lambda c: 0 if c.metadata.get('chunk_type') in ['overview', 'summary'] else 1)
        elif any(word in query_lower for word in ['code', 'diff', 'changes', 'modified']):
            # Prefer diff chunks
            filtered.sort(key=lambda c: 0 if c.metadata.get('chunk_type') == 'diff' else 1)
        
        return filtered
    
    def _rerank_chunks(self, query: str, chunks: List[Memory]) -> List[Memory]:
        """Rerank chunks using advanced scoring"""
        import re
        
        # Extract keywords from query
        keywords = set(re.findall(r'\b\w{3,}\b', query.lower()))
        
        for chunk in chunks:
            # Start with similarity score
            score = chunk.similarity or 0.5
            
            # Keyword density boost
            chunk_text_lower = chunk.content.lower()
            keyword_matches = sum(1 for kw in keywords if kw in chunk_text_lower)
            score += keyword_matches * 0.1
            
            # Chunk type boost
            chunk_type = chunk.metadata.get('chunk_type', '')
            if 'commit' in query.lower() and chunk_type == 'overview':
                score += 0.2
            elif 'change' in query.lower() and chunk_type == 'diff':
                score += 0.2
            elif 'file' in query.lower() and chunk_type == 'file_change':
                score += 0.15
            
            # Recency boost (newer = better)
            try:
                timestamp = self._parse_timestamp(chunk.metadata.get('date', ''))
                days_old = (datetime.now() - timestamp).days
                recency_score = max(0, (30 - days_old) / 30 * 0.1)
                score += recency_score
            except:
                pass
            
            # Store adjusted score
            chunk.adjusted_score = score
        
        # Sort by adjusted score
        chunks.sort(key=lambda c: getattr(c, 'adjusted_score', c.similarity or 0), reverse=True)
        
        return chunks
    
    def _build_optimized_prompt(self, query: str, chunks: List[Memory]) -> str:
        """Build focused prompt with only relevant context"""
        # Group chunks by commit
        chunks_by_commit = {}
        for chunk in chunks:
            sha = chunk.metadata.get('sha', 'unknown')
            if sha not in chunks_by_commit:
                chunks_by_commit[sha] = []
            chunks_by_commit[sha].append(chunk)
        
        # Build context sections
        context_parts = []
        
        for sha, commit_chunks in chunks_by_commit.items():
            context_parts.append(f"### Commit {sha}")
            
            # Add chunks in logical order
            chunk_types_order = ['overview', 'summary', 'file_change', 'diff']
            
            for chunk_type in chunk_types_order:
                type_chunks = [c for c in commit_chunks if c.metadata.get('chunk_type') == chunk_type]
                for chunk in type_chunks:
                    context_parts.append(chunk.content)
                    context_parts.append("")  # Empty line for readability
        
        context = "\n".join(context_parts)
        
        # Build final prompt
        prompt = f"""You are an AI assistant with access to the AI Memory Layer project's commit history.

Based on the following relevant commit information, answer the user's question accurately and concisely.

**User Question:** {query}

**Relevant Context:**
{context}

**Instructions:**
1. Answer the specific question asked
2. Reference specific commits, files, or code when relevant
3. Be concise but thorough
4. If the context doesn't contain enough information, say so
5. Focus on facts from the provided context

**Answer:**"""
        
        return prompt
    
    def _format_sources(self, chunks: List[Memory]) -> List[Dict[str, Any]]:
        """Format chunks as sources for response"""
        sources = []
        seen_shas = set()
        
        for chunk in chunks:
            sha = chunk.metadata.get('sha', 'unknown')
            
            # Include one source per commit
            if sha not in seen_shas:
                sources.append({
                    "sha": sha,
                    "author": chunk.metadata.get('author', 'unknown'),
                    "date": chunk.metadata.get('date', 'unknown'),
                    "chunk_type": chunk.metadata.get('chunk_type', 'unknown'),
                    "similarity": round(chunk.similarity or 0, 3) if chunk.similarity else None,
                    "preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                })
                seen_shas.add(sha)
        
        return sources
    
    def _parse_timestamp(self, date_str: str) -> datetime:
        """Parse various timestamp formats"""
        from dateutil import parser
        try:
            return parser.parse(date_str)
        except:
            return datetime.now()

# Integration into main.py:
"""
from optimized_query_handler import OptimizedQueryHandler

# Initialize optimized handler
query_handler = OptimizedQueryHandler(
    memory_query=memory_query,
    prompt_builder=prompt_builder,
    gpt_assistant=gpt_assistant,
    max_context_size=5,  # Only use top 5 chunks
    enable_reranking=True
)

@app.post("/query")
async def query(request: QueryRequest):
    result = await query_handler.handle_query(request.query)
    
    return {
        "response": result["response"],
        "sources": result["sources"],
        "processing_time": result["metrics"]["retrieval_time"],
        "debug": {
            "chunks_searched": result["metrics"]["chunks_searched"],
            "chunks_used": result["metrics"]["chunks_used"],
            "avg_similarity": result["metrics"]["avg_similarity"]
        }
    }
"""
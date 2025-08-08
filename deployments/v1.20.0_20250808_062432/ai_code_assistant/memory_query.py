#!/usr/bin/env python3
"""
Memory Query Service
====================

Handles semantic search and retrieval of memories from the vector store.
Provides intelligent ranking and context filtering for better AI responses.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from vector_store import VectorStore, Memory
from embedder import EmbeddingService

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Result from a memory query"""
    memories: List[Memory]
    query_embedding: List[float]
    total_found: int
    processing_time: float

class MemoryQuery:
    """Service for querying memories with semantic search"""
    
    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        logger.info("Memory query service initialized")
    
    async def search(self, query: str, max_results: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Memory]:
        """Search for memories similar to the query"""
        import time
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_text(query)
            
            # Search vector store
            memories = self.vector_store.search(query_embedding, max_results * 2)  # Get extra for filtering
            
            # Apply filters if provided
            if filters:
                memories = self._apply_filters(memories, filters)
            
            # Re-rank and limit results
            memories = self._rerank_memories(memories, query)[:max_results]
            
            processing_time = time.time() - start_time
            logger.info(f"Memory search completed in {processing_time:.3f}s, found {len(memories)} results")
            
            return memories
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
    
    def _apply_filters(self, memories: List[Memory], filters: Dict[str, Any]) -> List[Memory]:
        """Apply filters to memory results"""
        filtered_memories = []
        
        for memory in memories:
            # Check type filter
            if 'type' in filters:
                memory_type = memory.metadata.get('type')
                if memory_type != filters['type']:
                    continue
            
            # Check author filter
            if 'author' in filters:
                memory_author = memory.metadata.get('author')
                if memory_author != filters['author']:
                    continue
            
            # Check file filter
            if 'file' in filters:
                files_changed = memory.metadata.get('files_changed', [])
                if not any(filters['file'] in f for f in files_changed):
                    continue
            
            # Check recency filter (days)
            if 'max_age_days' in filters:
                from datetime import datetime, timedelta
                try:
                    memory_time = datetime.fromisoformat(memory.timestamp.replace('Z', '+00:00'))
                    cutoff_time = datetime.now() - timedelta(days=filters['max_age_days'])
                    if memory_time.replace(tzinfo=None) < cutoff_time:
                        continue
                except:
                    continue  # Skip if timestamp parsing fails
            
            filtered_memories.append(memory)
        
        logger.debug(f"Applied filters, {len(filtered_memories)}/{len(memories)} memories remain")
        return filtered_memories
    
    def _rerank_memories(self, memories: List[Memory], query: str) -> List[Memory]:
        """Re-rank memories based on relevance and recency"""
        import re
        from datetime import datetime
        
        for memory in memories:
            # Start with similarity score
            relevance_score = memory.similarity or 0.5
            
            # Boost score for exact keyword matches
            query_words = set(re.findall(r'\w+', query.lower()))
            content_words = set(re.findall(r'\w+', memory.content.lower()))
            keyword_overlap = len(query_words.intersection(content_words)) / max(len(query_words), 1)
            relevance_score += keyword_overlap * 0.2
            
            # Boost recent memories slightly
            try:
                memory_time = datetime.fromisoformat(memory.timestamp.replace('Z', '+00:00'))
                days_ago = (datetime.now() - memory_time.replace(tzinfo=None)).days
                recency_boost = max(0, (30 - days_ago) / 30 * 0.1)  # Boost memories from last 30 days
                relevance_score += recency_boost
            except:
                pass  # Skip recency boost if timestamp parsing fails
            
            # Boost commits over conversations for code-related queries
            code_keywords = {'function', 'class', 'method', 'bug', 'fix', 'implement', 'code', 'file'}
            if any(word in query.lower() for word in code_keywords):
                if memory.metadata.get('type') == 'commit':
                    relevance_score += 0.15
            
            # Store the final relevance score
            memory.similarity = float(relevance_score)
        
        # Sort by relevance score (descending)
        memories.sort(key=lambda m: m.similarity or 0, reverse=True)
        
        return memories
    
    async def search_by_commit(self, commit_sha: str) -> List[Memory]:
        """Search for memories related to a specific commit"""
        filters = {'type': 'commit'}
        memories = await self.search(f"commit {commit_sha}", max_results=10, filters=filters)
        
        # Also search for exact SHA match
        exact_matches = []
        for memory in self.vector_store.get_recent_memories(limit=100):
            if memory.metadata.get('sha', '').startswith(commit_sha):
                exact_matches.append(memory)
        
        # Combine and deduplicate
        all_memories = exact_matches + memories
        seen_ids = set()
        unique_memories = []
        
        for memory in all_memories:
            if memory.id not in seen_ids:
                unique_memories.append(memory)
                seen_ids.add(memory.id)
        
        return unique_memories[:5]
    
    async def search_by_file(self, filename: str) -> List[Memory]:
        """Search for memories related to a specific file"""
        return await self.search(f"file {filename}", max_results=10, filters={'file': filename})
    
    async def search_by_author(self, author: str, max_results: int = 10) -> List[Memory]:
        """Search for memories by a specific author"""
        return await self.search(f"author {author}", max_results=max_results, filters={'author': author})
    
    async def get_recent_context(self, hours: int = 24, max_results: int = 5) -> List[Memory]:
        """Get recent memories for context"""
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_memories = []
        
        for memory in self.vector_store.get_recent_memories(limit=50):
            try:
                memory_time = datetime.fromisoformat(memory.timestamp.replace('Z', '+00:00'))
                if memory_time.replace(tzinfo=None) > cutoff_time:
                    recent_memories.append(memory)
            except:
                continue  # Skip if timestamp parsing fails
        
        return recent_memories[:max_results]
    
    async def get_architectural_context(self, max_results: int = 3) -> List[Memory]:
        """Get memories related to system architecture"""
        architecture_queries = [
            "system architecture design structure",
            "component module integration",
            "API endpoint interface"
        ]
        
        all_memories = []
        for query in architecture_queries:
            memories = await self.search(query, max_results=2)
            all_memories.extend(memories)
        
        # Deduplicate and limit
        seen_ids = set()
        unique_memories = []
        
        for memory in all_memories:
            if memory.id not in seen_ids:
                unique_memories.append(memory)
                seen_ids.add(memory.id)
        
        return unique_memories[:max_results]
    
    async def search_with_context(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Enhanced search that includes contextual information"""
        # Main search
        main_results = await self.search(query, max_results=max_results)
        
        # Get recent context
        recent_context = await self.get_recent_context(hours=48, max_results=2)
        
        # Get architectural context if it's a technical query
        technical_keywords = {'how', 'why', 'architecture', 'design', 'implement', 'structure'}
        architectural_context = []
        if any(word in query.lower() for word in technical_keywords):
            architectural_context = await self.get_architectural_context(max_results=2)
        
        return {
            "main_results": main_results,
            "recent_context": recent_context,
            "architectural_context": architectural_context,
            "total_memories": len(main_results) + len(recent_context) + len(architectural_context)
        }
    
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """Get query suggestions based on available memories"""
        suggestions = []
        
        # Common query patterns
        common_queries = [
            "What changed in the latest commit?",
            "How does the memory engine work?",
            "Show me recent bug fixes",
            "What are the main system components?",
            "How is the API structured?",
            "What tests should I write?",
            "Show me performance improvements",
            "What's the current architecture?"
        ]
        
        # Filter suggestions based on partial query
        if partial_query:
            suggestions = [q for q in common_queries if partial_query.lower() in q.lower()]
        else:
            suggestions = common_queries[:5]
        
        return suggestions
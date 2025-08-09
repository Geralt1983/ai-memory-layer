#!/usr/bin/env python3
"""
Optimized Memory Loader with Intelligent Chunking
================================================

Splits commit summaries into semantic chunks for better retrieval.
"""

import re
import hashlib
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class CommitChunk:
    """Represents a semantic chunk from a commit"""
    chunk_id: str
    commit_sha: str
    chunk_type: str  # 'overview', 'summary', 'diff', 'file_change'
    content: str
    metadata: Dict[str, Any]
    
    def to_memory_content(self) -> str:
        """Format chunk for embedding"""
        if self.chunk_type == 'overview':
            return f"Commit {self.commit_sha}: {self.content}"
        elif self.chunk_type == 'summary':
            return f"Summary of {self.commit_sha}: {self.content}"
        elif self.chunk_type == 'diff':
            file_path = self.metadata.get('file_path', 'unknown')
            return f"Code changes in {file_path} (commit {self.commit_sha}):\n{self.content}"
        elif self.chunk_type == 'file_change':
            return f"Files changed in {self.commit_sha}: {self.content}"
        return self.content

class OptimizedCommitProcessor:
    """Processes commit files into semantic chunks"""
    
    def __init__(self, max_chunk_size: int = 1500):
        self.max_chunk_size = max_chunk_size
        
    def process_commit_file(self, file_path: Path) -> List[CommitChunk]:
        """Process a single commit .md file into chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract commit SHA from filename or content
            sha_match = re.search(r'([a-f0-9]{8,40})', file_path.stem)
            commit_sha = sha_match.group(1) if sha_match else 'unknown'
            
            chunks = []
            
            # 1. Extract and chunk the overview section
            overview_chunk = self._extract_overview(content, commit_sha)
            if overview_chunk:
                chunks.append(overview_chunk)
            
            # 2. Extract and chunk the summary
            summary_chunks = self._extract_summary_chunks(content, commit_sha)
            chunks.extend(summary_chunks)
            
            # 3. Extract and chunk individual file changes
            file_chunks = self._extract_file_chunks(content, commit_sha)
            chunks.extend(file_chunks)
            
            # 4. Extract and chunk diffs
            diff_chunks = self._extract_diff_chunks(content, commit_sha)
            chunks.extend(diff_chunks)
            
            logger.info(f"Processed {file_path.name} into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return []
    
    def _extract_overview(self, content: str, commit_sha: str) -> CommitChunk:
        """Extract commit overview (title + basic info)"""
        lines = content.split('\n')
        overview_lines = []
        
        for line in lines[:10]:  # First 10 lines typically contain overview
            if line.strip():
                overview_lines.append(line)
            if len('\n'.join(overview_lines)) > 300:
                break
        
        if overview_lines:
            overview_content = '\n'.join(overview_lines)
            chunk_id = self._generate_chunk_id(commit_sha, 'overview', overview_content)
            
            # Extract metadata
            author_match = re.search(r'Author:\s*(.+)', overview_content)
            date_match = re.search(r'Date:\s*(.+)', overview_content)
            
            return CommitChunk(
                chunk_id=chunk_id,
                commit_sha=commit_sha,
                chunk_type='overview',
                content=overview_content,
                metadata={
                    'author': author_match.group(1) if author_match else 'unknown',
                    'date': date_match.group(1) if date_match else 'unknown',
                    'chunk_index': 0
                }
            )
        return None
    
    def _extract_summary_chunks(self, content: str, commit_sha: str) -> List[CommitChunk]:
        """Extract summary section and chunk if needed"""
        chunks = []
        
        # Find summary section
        summary_match = re.search(r'##\s*(?:Summary|Description|Changes)(.*?)(?=##|\Z)', 
                                 content, re.IGNORECASE | re.DOTALL)
        
        if summary_match:
            summary_text = summary_match.group(1).strip()
            
            # Split into paragraphs or bullet points
            paragraphs = re.split(r'\n\s*\n', summary_text)
            
            current_chunk = []
            current_size = 0
            
            for para in paragraphs:
                para_size = len(para)
                
                if current_size + para_size > self.max_chunk_size and current_chunk:
                    # Save current chunk
                    chunk_content = '\n\n'.join(current_chunk)
                    chunk_id = self._generate_chunk_id(commit_sha, 'summary', chunk_content)
                    
                    chunks.append(CommitChunk(
                        chunk_id=chunk_id,
                        commit_sha=commit_sha,
                        chunk_type='summary',
                        content=chunk_content,
                        metadata={'chunk_index': len(chunks)}
                    ))
                    
                    current_chunk = [para]
                    current_size = para_size
                else:
                    current_chunk.append(para)
                    current_size += para_size
            
            # Add remaining content
            if current_chunk:
                chunk_content = '\n\n'.join(current_chunk)
                chunk_id = self._generate_chunk_id(commit_sha, 'summary', chunk_content)
                
                chunks.append(CommitChunk(
                    chunk_id=chunk_id,
                    commit_sha=commit_sha,
                    chunk_type='summary',
                    content=chunk_content,
                    metadata={'chunk_index': len(chunks)}
                ))
        
        return chunks
    
    def _extract_file_chunks(self, content: str, commit_sha: str) -> List[CommitChunk]:
        """Extract information about changed files"""
        chunks = []
        
        # Find files changed section
        files_match = re.search(r'##\s*(?:Files Changed|Changed Files)(.*?)(?=##|\Z)', 
                               content, re.IGNORECASE | re.DOTALL)
        
        if files_match:
            files_text = files_match.group(1).strip()
            
            # Extract file paths
            file_paths = re.findall(r'[•\-\*]\s*(.+\.(?:py|js|ts|md|yaml|json|txt|sh))', files_text)
            
            if file_paths:
                # Group files by type
                files_by_type = {}
                for path in file_paths:
                    ext = path.split('.')[-1]
                    if ext not in files_by_type:
                        files_by_type[ext] = []
                    files_by_type[ext].append(path)
                
                # Create chunks by file type
                for ext, paths in files_by_type.items():
                    chunk_content = f"{ext.upper()} files changed:\n" + '\n'.join(f"- {p}" for p in paths)
                    chunk_id = self._generate_chunk_id(commit_sha, 'file_change', chunk_content)
                    
                    chunks.append(CommitChunk(
                        chunk_id=chunk_id,
                        commit_sha=commit_sha,
                        chunk_type='file_change',
                        content=chunk_content,
                        metadata={
                            'file_type': ext,
                            'file_count': len(paths),
                            'files': paths
                        }
                    ))
        
        return chunks
    
    def _extract_diff_chunks(self, content: str, commit_sha: str) -> List[CommitChunk]:
        """Extract and chunk code diffs"""
        chunks = []
        
        # Find all diff blocks
        diff_pattern = r'```(?:diff|patch)\n(.*?)```'
        diffs = re.findall(diff_pattern, content, re.DOTALL)
        
        for i, diff in enumerate(diffs):
            # Try to extract file path from diff
            file_match = re.search(r'(?:---|\+\+\+)\s+(?:a/|b/)?(.+)', diff)
            file_path = file_match.group(1) if file_match else f'diff_{i}'
            
            # Split large diffs into chunks
            if len(diff) > self.max_chunk_size:
                # Split by hunks (@@)
                hunks = re.split(r'(?=@@)', diff)
                
                for j, hunk in enumerate(hunks):
                    if hunk.strip():
                        chunk_id = self._generate_chunk_id(commit_sha, f'diff_{i}_{j}', hunk)
                        
                        chunks.append(CommitChunk(
                            chunk_id=chunk_id,
                            commit_sha=commit_sha,
                            chunk_type='diff',
                            content=hunk,
                            metadata={
                                'file_path': file_path,
                                'diff_index': i,
                                'hunk_index': j
                            }
                        ))
            else:
                chunk_id = self._generate_chunk_id(commit_sha, f'diff_{i}', diff)
                
                chunks.append(CommitChunk(
                    chunk_id=chunk_id,
                    commit_sha=commit_sha,
                    chunk_type='diff',
                    content=diff,
                    metadata={
                        'file_path': file_path,
                        'diff_index': i
                    }
                ))
        
        return chunks
    
    def _generate_chunk_id(self, commit_sha: str, chunk_type: str, content: str) -> str:
        """Generate unique chunk ID"""
        hash_input = f"{commit_sha}:{chunk_type}:{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

async def process_all_commits(sync_dir: Path, vector_store, embedding_service) -> int:
    """Process all commit files and store chunks in vector store"""
    processor = OptimizedCommitProcessor()
    total_chunks = 0
    
    # Get all .md files
    md_files = list(sync_dir.glob("*.md"))
    logger.info(f"Found {len(md_files)} commit files to process")
    
    for md_file in md_files:
        # Check if already processed
        if vector_store.is_processed(str(md_file)):
            logger.debug(f"Skipping already processed: {md_file}")
            continue
        
        # Process into chunks
        chunks = processor.process_commit_file(md_file)
        
        # Store each chunk
        for chunk in chunks:
            try:
                # Generate embedding
                embedding = await embedding_service.embed_text(chunk.to_memory_content())
                
                # Prepare metadata
                metadata = {
                    'type': 'commit',
                    'sha': chunk.commit_sha,
                    'chunk_type': chunk.chunk_type,
                    'chunk_id': chunk.chunk_id,
                    'source_file': md_file.name,
                    **chunk.metadata
                }
                
                # Add to vector store
                vector_store.add_memory(
                    content=chunk.to_memory_content(),
                    embedding=embedding,
                    metadata=metadata
                )
                
                total_chunks += 1
                
            except Exception as e:
                logger.error(f"Failed to process chunk {chunk.chunk_id}: {e}")
        
        # Mark file as processed
        vector_store.mark_processed(str(md_file))
        logger.info(f"Processed {md_file.name} into {len(chunks)} chunks")
    
    return total_chunks

# Example usage in main.py:
"""
# Replace the simple file processing with:
from optimized_memory_loader import process_all_commits

@app.on_event("startup")
async def startup():
    global total_chunks
    total_chunks = await process_all_commits(SYNC_DIR, vector_store, embedding_service)
    logger.info(f"Loaded {total_chunks} semantic chunks")
"""
"""
Pydantic models for API request/response schemas
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class MemoryCreate(BaseModel):
    """Schema for creating a new memory"""
    content: str = Field(..., description="The memory content", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata")


class MemoryResponse(BaseModel):
    """Schema for memory response"""
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    relevance_score: float = 0.0


class ChatRequest(BaseModel):
    """Schema for chat requests"""
    message: str = Field(..., description="User message", min_length=1)
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    include_recent: int = Field(5, ge=0, le=20, description="Number of recent memories to include")
    include_relevant: int = Field(5, ge=0, le=20, description="Number of relevant memories to include")
    remember_response: bool = Field(True, description="Whether to store the conversation in memory")


class ChatResponse(BaseModel):
    """Schema for chat responses"""
    response: str
    context_used: Optional[str] = None


class SearchRequest(BaseModel):
    """Schema for memory search requests"""
    query: str = Field(..., description="Search query", min_length=1)
    k: int = Field(5, ge=1, le=50, description="Number of results to return")


class SearchResponse(BaseModel):
    """Schema for search results"""
    memories: List[MemoryResponse]
    total_count: int


class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str
    timestamp: datetime
    memory_count: int
    vector_store_type: Optional[str] = None


class ErrorResponse(BaseModel):
    """Schema for error responses"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
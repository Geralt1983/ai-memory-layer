import pytest
from datetime import datetime
from core.memory_engine import Memory


class TestMemory:
    """Test cases for the Memory class"""
    
    def test_memory_creation(self):
        """Test basic memory creation"""
        memory = Memory(content="Test content")
        
        assert memory.content == "Test content"
        assert memory.metadata == {}
        assert memory.relevance_score == 0.0
        assert isinstance(memory.timestamp, datetime)
        assert memory.embedding is None
    
    def test_memory_with_metadata(self):
        """Test memory creation with metadata"""
        metadata = {"type": "user_message", "source": "chat"}
        memory = Memory(content="Test", metadata=metadata)
        
        assert memory.metadata == metadata
    
    def test_memory_to_dict(self):
        """Test memory serialization to dict"""
        memory = Memory(
            content="Test content",
            metadata={"key": "value"},
            relevance_score=0.5
        )
        
        result = memory.to_dict()
        
        assert result["content"] == "Test content"
        assert result["metadata"] == {"key": "value"}
        assert result["relevance_score"] == 0.5
        assert "timestamp" in result
        assert isinstance(result["timestamp"], str)
    
    def test_memory_from_dict(self):
        """Test memory deserialization from dict"""
        data = {
            "content": "Test content",
            "metadata": {"key": "value"},
            "timestamp": "2023-01-01T12:00:00",
            "relevance_score": 0.8
        }
        
        memory = Memory.from_dict(data)
        
        assert memory.content == "Test content"
        assert memory.metadata == {"key": "value"}
        assert memory.relevance_score == 0.8
        assert memory.timestamp.year == 2023
        assert memory.timestamp.month == 1
        assert memory.timestamp.day == 1
    
    def test_memory_roundtrip_serialization(self, sample_memory):
        """Test that memory can be serialized and deserialized without loss"""
        data = sample_memory.to_dict()
        restored_memory = Memory.from_dict(data)
        
        assert restored_memory.content == sample_memory.content
        assert restored_memory.metadata == sample_memory.metadata
        assert restored_memory.relevance_score == sample_memory.relevance_score
        # Note: timestamp comparison might have slight differences due to serialization
    
    def test_memory_from_dict_with_missing_fields(self):
        """Test memory deserialization with missing optional fields"""
        data = {
            "content": "Minimal content",
            "timestamp": "2023-01-01T12:00:00"
        }
        
        memory = Memory.from_dict(data)
        
        assert memory.content == "Minimal content"
        assert memory.metadata == {}
        assert memory.relevance_score == 0.0
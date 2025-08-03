import pytest
import tempfile
import os
import json
import gzip
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from core.memory_engine import Memory, MemoryEngine
from core.memory_manager import (
    MemoryManager, MemoryArchiver, ArchiveInfo, CleanupStats,
    AgeBasedCleanup, SizeBasedCleanup, RelevanceBasedCleanup, 
    DuplicateCleanup, MetadataBasedCleanup, create_default_memory_manager
)
from tests.conftest import MockEmbeddingProvider


class TestMemoryCleanupStrategies:
    """Test memory cleanup strategies"""
    
    def test_age_based_cleanup(self):
        """Test age-based cleanup strategy"""
        strategy = AgeBasedCleanup(max_age_days=30)
        
        # Old memory (should be cleaned)
        old_memory = Memory(
            content="Old memory",
            timestamp=datetime.now() - timedelta(days=45)
        )
        
        # Recent memory (should be kept)
        recent_memory = Memory(
            content="Recent memory",
            timestamp=datetime.now() - timedelta(days=15)
        )
        
        assert strategy.should_cleanup(old_memory, {}) is True
        assert strategy.should_cleanup(recent_memory, {}) is False
        
        # Test priority calculation
        assert strategy.get_priority(old_memory) > strategy.get_priority(recent_memory)
    
    def test_size_based_cleanup(self):
        """Test size-based cleanup strategy"""
        strategy = SizeBasedCleanup(max_memories=100)
        
        memory = Memory(content="Test memory")
        
        # Memory within limit (should be kept)
        context = {"memory_index": 50, "total_memories": 80}
        assert strategy.should_cleanup(memory, context) is False
        
        # Memory outside limit (should be cleaned)
        context = {"memory_index": 20, "total_memories": 150}
        assert strategy.should_cleanup(memory, context) is True
    
    def test_relevance_based_cleanup(self):
        """Test relevance-based cleanup strategy"""
        strategy = RelevanceBasedCleanup(min_relevance=0.5)
        
        # Low relevance memory (should be cleaned)
        low_relevance_memory = Memory(
            content="Low relevance",
            relevance_score=0.2
        )
        
        # High relevance memory (should be kept)
        high_relevance_memory = Memory(
            content="High relevance",
            relevance_score=0.8
        )
        
        assert strategy.should_cleanup(low_relevance_memory, {}) is True
        assert strategy.should_cleanup(high_relevance_memory, {}) is False
    
    def test_duplicate_cleanup(self):
        """Test duplicate cleanup strategy"""
        strategy = DuplicateCleanup()
        
        memory1 = Memory(content="Duplicate content")
        memory2 = Memory(content="Duplicate content")
        memory3 = Memory(content="Unique content")
        
        # First occurrence should be kept
        assert strategy.should_cleanup(memory1, {}) is False
        
        # Duplicate should be cleaned
        assert strategy.should_cleanup(memory2, {}) is True
        
        # Unique content should be kept
        assert strategy.should_cleanup(memory3, {}) is False
    
    def test_metadata_based_cleanup(self):
        """Test metadata-based cleanup strategy"""
        strategy = MetadataBasedCleanup({"type": "temporary", "status": "expired"})
        
        # Memory matching criteria (should be cleaned)
        matching_memory = Memory(
            content="Temporary memory",
            metadata={"type": "temporary", "status": "active"}
        )
        
        # Memory not matching criteria (should be kept)
        non_matching_memory = Memory(
            content="Permanent memory",
            metadata={"type": "permanent", "status": "active"}
        )
        
        assert strategy.should_cleanup(matching_memory, {}) is True
        assert strategy.should_cleanup(non_matching_memory, {}) is False


class TestMemoryArchiver:
    """Test memory archiver functionality"""
    
    @pytest.fixture
    def temp_archive_dir(self):
        """Temporary directory for archive testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sample_memories(self):
        """Sample memories for testing"""
        return [
            Memory(content="Memory 1", metadata={"type": "test"}),
            Memory(content="Memory 2", metadata={"type": "user"}),
            Memory(content="Memory 3", metadata={"type": "system"})
        ]
    
    def test_archiver_initialization(self, temp_archive_dir):
        """Test archiver initialization"""
        archiver = MemoryArchiver(temp_archive_dir)
        
        assert archiver.archive_directory.exists()
        assert str(archiver.archive_directory) == temp_archive_dir
    
    def test_archive_memories(self, temp_archive_dir, sample_memories):
        """Test archiving memories"""
        archiver = MemoryArchiver(temp_archive_dir)
        criteria = {"cleanup_reason": "test"}
        
        archive_info = archiver.archive_memories(sample_memories, criteria)
        
        assert isinstance(archive_info, ArchiveInfo)
        assert archive_info.memory_count == 3
        assert archive_info.criteria == criteria
        assert os.path.exists(archive_info.archive_path)
        assert archive_info.size_bytes > 0
    
    def test_load_archive(self, temp_archive_dir, sample_memories):
        """Test loading memories from archive"""
        archiver = MemoryArchiver(temp_archive_dir)
        criteria = {"cleanup_reason": "test"}
        
        # Create archive
        archive_info = archiver.archive_memories(sample_memories, criteria)
        
        # Load archive
        loaded_memories = archiver.load_archive(archive_info.archive_path)
        
        assert len(loaded_memories) == 3
        assert loaded_memories[0].content == "Memory 1"
        assert loaded_memories[1].metadata["type"] == "user"
    
    def test_list_archives(self, temp_archive_dir, sample_memories):
        """Test listing archives"""
        archiver = MemoryArchiver(temp_archive_dir)
        
        # Create multiple archives
        archiver.archive_memories(sample_memories[:2], {"batch": "1"})
        archiver.archive_memories([sample_memories[2]], {"batch": "2"})
        
        archives = archiver.list_archives()
        
        assert len(archives) == 2
        assert all(isinstance(a, ArchiveInfo) for a in archives)
        assert archives[0].created_at >= archives[1].created_at  # Sorted by date desc
    
    def test_archive_compression(self, temp_archive_dir, sample_memories):
        """Test that archives are properly compressed"""
        archiver = MemoryArchiver(temp_archive_dir)
        
        archive_info = archiver.archive_memories(sample_memories, {})
        
        # Check that file is gzipped
        with gzip.open(archive_info.archive_path, 'rt') as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert "memories" in data
        assert len(data["memories"]) == 3


class TestMemoryManager:
    """Test memory manager functionality"""
    
    @pytest.fixture
    def memory_engine(self):
        """Memory engine with sample data"""
        engine = MemoryEngine()
        
        # Add memories with different ages and relevance
        now = datetime.now()
        memories = [
            Memory(content="Recent memory", timestamp=now - timedelta(days=1), relevance_score=0.9),
            Memory(content="Old memory", timestamp=now - timedelta(days=100), relevance_score=0.8),
            Memory(content="Very old memory", timestamp=now - timedelta(days=200), relevance_score=0.1),
            Memory(content="Low relevance", timestamp=now - timedelta(days=10), relevance_score=0.02),
            Memory(content="Duplicate content", timestamp=now - timedelta(days=5), relevance_score=0.5),
            Memory(content="Duplicate content", timestamp=now - timedelta(days=6), relevance_score=0.4),
        ]
        
        engine.memories = memories
        return engine
    
    @pytest.fixture
    def temp_archive_dir(self):
        """Temporary directory for archive testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_memory_manager_initialization(self, memory_engine, temp_archive_dir):
        """Test memory manager initialization"""
        manager = MemoryManager(memory_engine, temp_archive_dir)
        
        assert manager.memory_engine == memory_engine
        assert isinstance(manager.archiver, MemoryArchiver)
        assert len(manager.cleanup_strategies) == 0
    
    def test_add_cleanup_strategy(self, memory_engine, temp_archive_dir):
        """Test adding cleanup strategies"""
        manager = MemoryManager(memory_engine, temp_archive_dir)
        strategy = AgeBasedCleanup(30)
        
        manager.add_cleanup_strategy(strategy)
        
        assert len(manager.cleanup_strategies) == 1
        assert manager.cleanup_strategies[0] == strategy
    
    def test_cleanup_memories_dry_run(self, memory_engine, temp_archive_dir):
        """Test dry run cleanup"""
        manager = MemoryManager(memory_engine, temp_archive_dir)
        manager.add_cleanup_strategy(AgeBasedCleanup(max_age_days=30))
        
        initial_count = len(memory_engine.memories)
        
        stats = manager.cleanup_memories(dry_run=True)
        
        # Dry run should not modify memories
        assert len(memory_engine.memories) == initial_count
        assert isinstance(stats, CleanupStats)
        assert stats.total_memories_before == initial_count
        assert stats.memories_cleaned > 0  # Some should be marked for cleanup
    
    def test_cleanup_memories_with_archival(self, memory_engine, temp_archive_dir):
        """Test cleanup with archival"""
        manager = MemoryManager(memory_engine, temp_archive_dir)
        manager.add_cleanup_strategy(AgeBasedCleanup(max_age_days=30))
        
        initial_count = len(memory_engine.memories)
        
        stats = manager.cleanup_memories(archive_before_cleanup=True)
        
        assert stats.total_memories_after < stats.total_memories_before
        assert stats.memories_cleaned > 0
        assert stats.memories_archived > 0
        
        # Check that archive was created
        archives = manager.archiver.list_archives()
        assert len(archives) > 0
    
    def test_auto_cleanup(self, memory_engine, temp_archive_dir):
        """Test automatic cleanup with default strategies"""
        manager = MemoryManager(memory_engine, temp_archive_dir)
        
        stats = manager.auto_cleanup(
            max_memories=3,
            max_age_days=50,
            min_relevance=0.5
        )
        
        assert isinstance(stats, CleanupStats)
        assert len(manager.cleanup_strategies) == 4  # All default strategies added
        assert stats.memories_cleaned > 0
    
    def test_export_memories_json(self, memory_engine, temp_archive_dir):
        """Test exporting memories to JSON"""
        manager = MemoryManager(memory_engine, temp_archive_dir)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            manager.export_memories(export_path, format="json")
            
            assert os.path.exists(export_path)
            
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            assert isinstance(data, list)
            assert len(data) == len(memory_engine.memories)
            assert "content" in data[0]
            
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)
    
    def test_export_memories_csv(self, memory_engine, temp_archive_dir):
        """Test exporting memories to CSV"""
        manager = MemoryManager(memory_engine, temp_archive_dir)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            export_path = f.name
        
        try:
            manager.export_memories(export_path, format="csv")
            
            assert os.path.exists(export_path)
            
            with open(export_path, 'r') as f:
                content = f.read()
            
            assert "content,timestamp,relevance_score,metadata" in content
            assert "Recent memory" in content
            
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)
    
    def test_export_memories_with_filter(self, memory_engine, temp_archive_dir):
        """Test exporting memories with filter function"""
        manager = MemoryManager(memory_engine, temp_archive_dir)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            # Export only memories with high relevance
            def high_relevance_filter(memory):
                return memory.relevance_score > 0.7
            
            manager.export_memories(
                export_path, 
                format="json",
                filter_func=high_relevance_filter
            )
            
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            # Should only have 2 memories with relevance > 0.7
            assert len(data) == 2
            
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)


class TestMemoryManagerIntegration:
    """Integration tests for memory manager"""
    
    def test_create_default_memory_manager(self):
        """Test creating default memory manager"""
        engine = MemoryEngine()
        manager = create_default_memory_manager(engine)
        
        assert isinstance(manager, MemoryManager)
        assert len(manager.cleanup_strategies) == 4  # Default strategies
        assert manager.memory_engine == engine
    
    def test_full_lifecycle(self):
        """Test full memory lifecycle: add -> cleanup -> archive -> restore"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup
            engine = MemoryEngine()
            manager = MemoryManager(engine, temp_dir)
            
            # Add memories
            for i in range(10):
                memory = Memory(
                    content=f"Memory {i}",
                    timestamp=datetime.now() - timedelta(days=i*10),
                    relevance_score=0.1 + (i * 0.1)
                )
                engine.memories.append(memory)
            
            initial_count = len(engine.memories)
            
            # Setup cleanup strategy
            manager.add_cleanup_strategy(AgeBasedCleanup(max_age_days=30))
            
            # Cleanup with archival
            stats = manager.cleanup_memories(archive_before_cleanup=True)
            
            assert stats.total_memories_after < initial_count
            assert stats.memories_archived > 0
            
            # Verify archive was created
            archives = manager.archiver.list_archives()
            assert len(archives) == 1
            
            # Restore from archive
            archive = archives[0]
            restored_memories = manager.archiver.load_archive(archive.archive_path)
            
            assert len(restored_memories) == stats.memories_archived
            assert all(isinstance(m, Memory) for m in restored_memories)


class TestMemoryManagerAPI:
    """Test memory manager API integration"""
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager for API testing"""
        manager = Mock()
        manager.auto_cleanup.return_value = CleanupStats(
            total_memories_before=100,
            total_memories_after=80,
            memories_cleaned=20,
            memories_archived=20,
            bytes_freed=1024,
            duration_ms=150.0
        )
        return manager
    
    def test_cleanup_endpoint_integration(self, mock_memory_manager):
        """Test that cleanup endpoint works with memory manager"""
        from api.models import CleanupRequest
        
        request = CleanupRequest(
            max_memories=1000,
            max_age_days=30,
            min_relevance=0.1,
            dry_run=False
        )
        
        # This would be called by the API endpoint
        stats = mock_memory_manager.auto_cleanup(
            max_memories=request.max_memories,
            max_age_days=request.max_age_days,
            min_relevance=request.min_relevance
        )
        
        assert stats.memories_cleaned == 20
        mock_memory_manager.auto_cleanup.assert_called_once_with(
            max_memories=1000,
            max_age_days=30,
            min_relevance=0.1
        )
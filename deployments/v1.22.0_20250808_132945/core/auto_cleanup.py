"""
Automated Memory Cleanup System - Manages memory lifecycle and optimization
"""

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json

from core.logging_config import get_logger


class CleanupStrategy(Enum):
    """Cleanup strategies for memory management"""
    AGE_BASED = "age_based"
    SIZE_BASED = "size_based"
    RELEVANCE_BASED = "relevance_based"
    ACTIVITY_BASED = "activity_based"
    COMBINED = "combined"


@dataclass
class CleanupConfig:
    """Configuration for automated cleanup"""
    enabled: bool = True
    interval_hours: int = 24  # Run cleanup every 24 hours
    max_memory_count: int = 10000  # Maximum memories to keep
    max_age_days: int = 90  # Remove memories older than 90 days
    min_relevance_score: float = 0.3  # Remove memories with low relevance
    inactive_days: int = 30  # Consider memory inactive after 30 days
    archive_before_delete: bool = True  # Archive memories before deletion
    archive_path: str = "data/archives"
    strategy: CleanupStrategy = CleanupStrategy.COMBINED
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary"""
        strategy = config_dict.get('strategy', 'combined')
        if isinstance(strategy, str):
            strategy = CleanupStrategy(strategy)
        config_dict['strategy'] = strategy
        return cls(**config_dict)
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'enabled': self.enabled,
            'interval_hours': self.interval_hours,
            'max_memory_count': self.max_memory_count,
            'max_age_days': self.max_age_days,
            'min_relevance_score': self.min_relevance_score,
            'inactive_days': self.inactive_days,
            'archive_before_delete': self.archive_before_delete,
            'archive_path': self.archive_path,
            'strategy': self.strategy.value
        }


class AutoMemoryCleanup:
    """
    Automated memory cleanup system with configurable strategies
    """
    
    def __init__(
        self,
        memory_engine,
        config: Optional[CleanupConfig] = None,
        config_path: Optional[str] = "data/cleanup_config.json"
    ):
        """
        Initialize automated cleanup system
        
        Args:
            memory_engine: The memory engine to manage
            config: Cleanup configuration
            config_path: Path to save/load configuration
        """
        self.memory_engine = memory_engine
        self.config = config or CleanupConfig()
        self.config_path = config_path
        self.logger = get_logger("auto_cleanup")
        
        # Cleanup statistics
        self.stats = {
            'last_cleanup': None,
            'total_cleaned': 0,
            'total_archived': 0,
            'cleanup_runs': 0
        }
        
        # Background thread management
        self._cleanup_thread = None
        self._stop_event = threading.Event()
        
        # Load configuration if exists
        self._load_config()
        
        # Start automated cleanup if enabled
        if self.config.enabled:
            self.start()
        
        self.logger.info(
            "Auto cleanup initialized",
            extra={
                "config": self.config.to_dict(),
                "memory_count": len(memory_engine.memories)
            }
        )
    
    def start(self):
        """Start automated cleanup background thread"""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self.logger.warning("Cleanup thread already running")
            return
        
        self._stop_event.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True
        )
        self._cleanup_thread.start()
        self.logger.info("Automated cleanup started")
    
    def stop(self):
        """Stop automated cleanup background thread"""
        self._stop_event.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        self.logger.info("Automated cleanup stopped")
    
    def _cleanup_worker(self):
        """Background worker for automated cleanup"""
        interval_seconds = self.config.interval_hours * 3600
        
        while not self._stop_event.is_set():
            try:
                # Run cleanup
                self.run_cleanup()
                
                # Save statistics
                self._save_stats()
                
            except Exception as e:
                self.logger.error(
                    "Error in cleanup worker",
                    extra={"error": str(e)},
                    exc_info=True
                )
            
            # Wait for next interval
            self._stop_event.wait(interval_seconds)
    
    def run_cleanup(self, dry_run: bool = False) -> Dict:
        """
        Run memory cleanup based on configured strategy
        
        Args:
            dry_run: If True, only simulate cleanup without actual deletion
            
        Returns:
            Cleanup results dictionary
        """
        self.logger.info(
            "Starting memory cleanup",
            extra={
                "strategy": self.config.strategy.value,
                "dry_run": dry_run,
                "current_memories": len(self.memory_engine.memories)
            }
        )
        
        results = {
            'strategy': self.config.strategy.value,
            'dry_run': dry_run,
            'before_count': len(self.memory_engine.memories),
            'cleaned': 0,
            'archived': 0,
            'errors': []
        }
        
        try:
            # Get memories to clean based on strategy
            memories_to_clean = self._get_memories_to_clean()
            
            if memories_to_clean:
                # Archive if configured
                if self.config.archive_before_delete and not dry_run:
                    archived = self._archive_memories(memories_to_clean)
                    results['archived'] = archived
                    self.stats['total_archived'] += archived
                
                # Clean memories
                if not dry_run:
                    for memory in memories_to_clean:
                        try:
                            self.memory_engine.delete_memory(memory.id)
                            results['cleaned'] += 1
                        except Exception as e:
                            results['errors'].append(str(e))
                else:
                    results['cleaned'] = len(memories_to_clean)
                
                # Update statistics
                if not dry_run:
                    self.stats['total_cleaned'] += results['cleaned']
                    self.stats['cleanup_runs'] += 1
                    self.stats['last_cleanup'] = datetime.now(timezone.utc).isoformat()
            
            results['after_count'] = len(self.memory_engine.memories)
            
            self.logger.info(
                "Cleanup completed",
                extra=results
            )
            
        except Exception as e:
            self.logger.error(
                "Cleanup failed",
                extra={"error": str(e)},
                exc_info=True
            )
            results['errors'].append(str(e))
        
        return results
    
    def _get_memories_to_clean(self) -> List:
        """
        Get list of memories to clean based on strategy
        
        Returns:
            List of memories to clean
        """
        memories_to_clean = []
        
        if self.config.strategy == CleanupStrategy.AGE_BASED:
            memories_to_clean = self._get_old_memories()
            
        elif self.config.strategy == CleanupStrategy.SIZE_BASED:
            memories_to_clean = self._get_excess_memories()
            
        elif self.config.strategy == CleanupStrategy.RELEVANCE_BASED:
            memories_to_clean = self._get_low_relevance_memories()
            
        elif self.config.strategy == CleanupStrategy.ACTIVITY_BASED:
            memories_to_clean = self._get_inactive_memories()
            
        elif self.config.strategy == CleanupStrategy.COMBINED:
            # Combined strategy: apply all criteria
            old_memories = set(self._get_old_memories())
            excess_memories = set(self._get_excess_memories())
            low_relevance = set(self._get_low_relevance_memories())
            inactive = set(self._get_inactive_memories())
            
            # Union of all strategies
            combined = old_memories | excess_memories | low_relevance | inactive
            memories_to_clean = list(combined)
        
        return memories_to_clean
    
    def _get_old_memories(self) -> List:
        """Get memories older than configured age"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config.max_age_days)
        return [
            m for m in self.memory_engine.memories
            if hasattr(m, 'timestamp') and m.timestamp < cutoff_date
        ]
    
    def _get_excess_memories(self) -> List:
        """Get memories exceeding max count (keep most recent)"""
        if len(self.memory_engine.memories) <= self.config.max_memory_count:
            return []
        
        # Sort by timestamp (newest first)
        sorted_memories = sorted(
            self.memory_engine.memories,
            key=lambda m: getattr(m, 'timestamp', datetime.min.replace(tzinfo=timezone.utc)),
            reverse=True
        )
        
        # Return memories beyond the limit
        return sorted_memories[self.config.max_memory_count:]
    
    def _get_low_relevance_memories(self) -> List:
        """Get memories with low relevance scores"""
        return [
            m for m in self.memory_engine.memories
            if hasattr(m, 'relevance_score') and 
            getattr(m, 'relevance_score', 1.0) < self.config.min_relevance_score
        ]
    
    def _get_inactive_memories(self) -> List:
        """Get memories that haven't been accessed recently"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config.inactive_days)
        inactive = []
        
        for memory in self.memory_engine.memories:
            # Check last accessed time if available
            if hasattr(memory, 'last_accessed'):
                if memory.last_accessed and memory.last_accessed < cutoff_date:
                    inactive.append(memory)
            # Otherwise check creation time
            elif hasattr(memory, 'timestamp'):
                if memory.timestamp < cutoff_date:
                    inactive.append(memory)
        
        return inactive
    
    def _archive_memories(self, memories: List) -> int:
        """
        Archive memories before deletion
        
        Args:
            memories: List of memories to archive
            
        Returns:
            Number of memories archived
        """
        try:
            # Create archive directory
            archive_dir = Path(self.config.archive_path)
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Create archive file with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            archive_file = archive_dir / f"archive_{timestamp}.json"
            
            # Serialize memories
            archive_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'count': len(memories),
                'memories': [
                    {
                        'id': getattr(m, 'id', None),
                        'content': m.content,
                        'timestamp': m.timestamp.isoformat() if hasattr(m, 'timestamp') else None,
                        'metadata': getattr(m, 'metadata', {})
                    }
                    for m in memories
                ]
            }
            
            # Save archive
            with open(archive_file, 'w') as f:
                json.dump(archive_data, f, indent=2)
            
            self.logger.info(
                f"Archived {len(memories)} memories",
                extra={"archive_file": str(archive_file)}
            )
            
            return len(memories)
            
        except Exception as e:
            self.logger.error(
                "Failed to archive memories",
                extra={"error": str(e)},
                exc_info=True
            )
            return 0
    
    def _load_config(self):
        """Load configuration from file"""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                self.config = CleanupConfig.from_dict(config_data)
                self.logger.info("Loaded cleanup configuration")
            except Exception as e:
                self.logger.error(f"Failed to load config: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        if self.config_path:
            try:
                Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_path, 'w') as f:
                    json.dump(self.config.to_dict(), f, indent=2)
                self.logger.info("Saved cleanup configuration")
            except Exception as e:
                self.logger.error(f"Failed to save config: {e}")
    
    def _save_stats(self):
        """Save cleanup statistics"""
        try:
            stats_file = Path(self.config_path).parent / "cleanup_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save stats: {e}")
    
    def get_stats(self) -> Dict:
        """Get cleanup statistics"""
        return {
            **self.stats,
            'config': self.config.to_dict(),
            'current_memories': len(self.memory_engine.memories)
        }
    
    def update_config(self, **kwargs):
        """Update cleanup configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.save_config()
        self.logger.info(
            "Configuration updated",
            extra={"changes": kwargs}
        )
        
        # Restart if enabled state changed
        if 'enabled' in kwargs:
            if kwargs['enabled']:
                self.start()
            else:
                self.stop()
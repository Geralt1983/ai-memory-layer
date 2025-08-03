"""
Memory cleanup and archival strategies for AI Memory Layer
"""

import os
import json
import gzip
import shutil
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib

from .memory_engine import Memory, MemoryEngine
from .logging_config import get_logger, monitor_performance, log_memory_operation


@dataclass
class ArchiveInfo:
    """Information about an archived memory set"""

    archive_path: str
    created_at: datetime
    memory_count: int
    size_bytes: int
    criteria: Dict[str, Any]


@dataclass
class CleanupStats:
    """Statistics from cleanup operations"""

    total_memories_before: int
    total_memories_after: int
    memories_cleaned: int
    memories_archived: int
    bytes_freed: int
    duration_ms: float


class MemoryCleanupStrategy:
    """Base class for memory cleanup strategies"""

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"cleanup.{name}")

    def should_cleanup(self, memory: Memory, context: Dict[str, Any]) -> bool:
        """Determine if a memory should be cleaned up"""
        raise NotImplementedError

    def get_priority(self, memory: Memory) -> float:
        """Get cleanup priority (higher = more likely to be cleaned)"""
        return 0.0


class AgeBasedCleanup(MemoryCleanupStrategy):
    """Clean up memories older than specified age"""

    def __init__(self, max_age_days: int = 30):
        super().__init__("age_based")
        self.max_age_days = max_age_days
        self.cutoff_date = datetime.now() - timedelta(days=max_age_days)

    def should_cleanup(self, memory: Memory, context: Dict[str, Any]) -> bool:
        return memory.timestamp < self.cutoff_date

    def get_priority(self, memory: Memory) -> float:
        age_days = (datetime.now() - memory.timestamp).days
        return max(0, age_days - self.max_age_days) / 10.0


class SizeBasedCleanup(MemoryCleanupStrategy):
    """Keep only N most recent memories"""

    def __init__(self, max_memories: int = 1000):
        super().__init__("size_based")
        self.max_memories = max_memories

    def should_cleanup(self, memory: Memory, context: Dict[str, Any]) -> bool:
        memory_index = context.get("memory_index", 0)
        total_memories = context.get("total_memories", 0)

        # Keep the most recent max_memories
        return memory_index < (total_memories - self.max_memories)

    def get_priority(self, memory: Memory) -> float:
        # Older memories have higher cleanup priority
        age_days = (datetime.now() - memory.timestamp).days
        return age_days / 100.0


class RelevanceBasedCleanup(MemoryCleanupStrategy):
    """Clean up memories with low relevance scores"""

    def __init__(self, min_relevance: float = 0.1):
        super().__init__("relevance_based")
        self.min_relevance = min_relevance

    def should_cleanup(self, memory: Memory, context: Dict[str, Any]) -> bool:
        return memory.relevance_score < self.min_relevance

    def get_priority(self, memory: Memory) -> float:
        return max(0, self.min_relevance - memory.relevance_score)


class DuplicateCleanup(MemoryCleanupStrategy):
    """Remove duplicate or very similar memories"""

    def __init__(self, similarity_threshold: float = 0.95):
        super().__init__("duplicate_based")
        self.similarity_threshold = similarity_threshold
        self.seen_hashes = set()
        self.content_hashes = {}

    def _content_hash(self, content: str) -> str:
        """Generate hash of memory content"""
        return hashlib.md5(content.lower().strip().encode()).hexdigest()

    def should_cleanup(self, memory: Memory, context: Dict[str, Any]) -> bool:
        content_hash = self._content_hash(memory.content)

        if content_hash in self.seen_hashes:
            return True

        self.seen_hashes.add(content_hash)
        self.content_hashes[content_hash] = memory
        return False

    def get_priority(self, memory: Memory) -> float:
        # Shorter content gets higher cleanup priority if it's a duplicate
        return 1.0 / max(1, len(memory.content) / 100)


class MetadataBasedCleanup(MemoryCleanupStrategy):
    """Clean up memories based on metadata criteria"""

    def __init__(self, criteria: Dict[str, Any]):
        super().__init__("metadata_based")
        self.criteria = criteria

    def should_cleanup(self, memory: Memory, context: Dict[str, Any]) -> bool:
        for key, value in self.criteria.items():
            if key in memory.metadata:
                if isinstance(value, list):
                    if memory.metadata[key] in value:
                        return True
                elif memory.metadata[key] == value:
                    return True
        return False

    def get_priority(self, memory: Memory) -> float:
        matches = sum(
            1
            for key, value in self.criteria.items()
            if key in memory.metadata and memory.metadata[key] == value
        )
        return matches / len(self.criteria)


class MemoryArchiver:
    """Handles archiving memories to compressed storage"""

    def __init__(self, archive_directory: str = "./archives"):
        self.archive_directory = Path(archive_directory)
        self.archive_directory.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("archiver")

    @monitor_performance("archive_memories")
    def archive_memories(
        self, memories: List[Memory], criteria: Dict[str, Any]
    ) -> ArchiveInfo:
        """Archive memories to compressed file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        criteria_hash = hashlib.md5(str(sorted(criteria.items())).encode()).hexdigest()[
            :8
        ]
        archive_filename = f"memories_{timestamp}_{criteria_hash}.json.gz"
        archive_path = self.archive_directory / archive_filename

        self.logger.info(
            "Starting memory archival",
            extra={
                "memory_count": len(memories),
                "archive_path": str(archive_path),
                "criteria": criteria,
            },
        )

        # Prepare archive data
        archive_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "memory_count": len(memories),
                "criteria": criteria,
                "version": "1.0",
            },
            "memories": [memory.to_dict() for memory in memories],
        }

        # Write compressed archive
        with gzip.open(archive_path, "wt", encoding="utf-8") as f:
            json.dump(archive_data, f, indent=2, default=str)

        # Get file size
        size_bytes = archive_path.stat().st_size

        archive_info = ArchiveInfo(
            archive_path=str(archive_path),
            created_at=datetime.now(),
            memory_count=len(memories),
            size_bytes=size_bytes,
            criteria=criteria,
        )

        log_memory_operation(
            "archive",
            memory_count=len(memories),
            archive_path=str(archive_path),
            size_bytes=size_bytes,
        )

        self.logger.info(
            "Memory archival completed",
            extra={
                "memory_count": len(memories),
                "archive_path": str(archive_path),
                "size_bytes": size_bytes,
                "compression_ratio": self._estimate_compression_ratio(
                    memories, size_bytes
                ),
            },
        )

        return archive_info

    def _estimate_compression_ratio(
        self, memories: List[Memory], compressed_size: int
    ) -> float:
        """Estimate compression ratio"""
        uncompressed_size = sum(
            len(json.dumps(memory.to_dict(), default=str)) for memory in memories
        )
        return compressed_size / max(1, uncompressed_size)

    def load_archive(self, archive_path: str) -> List[Memory]:
        """Load memories from archive file"""
        self.logger.info("Loading archive", extra={"archive_path": archive_path})

        try:
            with gzip.open(archive_path, "rt", encoding="utf-8") as f:
                archive_data = json.load(f)

            memories = [Memory.from_dict(data) for data in archive_data["memories"]]

            self.logger.info(
                "Archive loaded successfully",
                extra={"archive_path": archive_path, "memory_count": len(memories)},
            )

            return memories

        except Exception as e:
            self.logger.error(
                "Failed to load archive",
                extra={"archive_path": archive_path, "error": str(e)},
                exc_info=True,
            )
            raise

    def list_archives(self) -> List[ArchiveInfo]:
        """List all available archives"""
        archives = []

        for archive_file in self.archive_directory.glob("*.json.gz"):
            try:
                with gzip.open(archive_file, "rt", encoding="utf-8") as f:
                    archive_data = json.load(f)

                metadata = archive_data.get("metadata", {})
                archives.append(
                    ArchiveInfo(
                        archive_path=str(archive_file),
                        created_at=datetime.fromisoformat(
                            metadata.get("created_at", "1970-01-01T00:00:00")
                        ),
                        memory_count=metadata.get("memory_count", 0),
                        size_bytes=archive_file.stat().st_size,
                        criteria=metadata.get("criteria", {}),
                    )
                )

            except Exception as e:
                self.logger.warning(
                    "Failed to read archive metadata",
                    extra={"archive_path": str(archive_file), "error": str(e)},
                )

        return sorted(archives, key=lambda a: a.created_at, reverse=True)


class MemoryManager:
    """Main class for managing memory cleanup and archival"""

    def __init__(
        self, memory_engine: MemoryEngine, archive_directory: str = "./archives"
    ):
        self.memory_engine = memory_engine
        self.archiver = MemoryArchiver(archive_directory)
        self.cleanup_strategies: List[MemoryCleanupStrategy] = []
        self.logger = get_logger("memory_manager")

        self.logger.info(
            "Memory manager initialized", extra={"archive_directory": archive_directory}
        )

    def add_cleanup_strategy(self, strategy: MemoryCleanupStrategy):
        """Add a cleanup strategy"""
        self.cleanup_strategies.append(strategy)
        self.logger.info("Cleanup strategy added", extra={"strategy": strategy.name})

    @monitor_performance("cleanup_memories")
    def cleanup_memories(
        self, archive_before_cleanup: bool = True, dry_run: bool = False
    ) -> CleanupStats:
        """Clean up memories using configured strategies"""
        start_time = datetime.now()
        initial_count = len(self.memory_engine.memories)

        self.logger.info(
            "Starting memory cleanup",
            extra={
                "initial_memory_count": initial_count,
                "strategies": [s.name for s in self.cleanup_strategies],
                "archive_before_cleanup": archive_before_cleanup,
                "dry_run": dry_run,
            },
        )

        if not self.cleanup_strategies:
            self.logger.warning("No cleanup strategies configured")
            return CleanupStats(
                total_memories_before=initial_count,
                total_memories_after=initial_count,
                memories_cleaned=0,
                memories_archived=0,
                bytes_freed=0,
                duration_ms=0,
            )

        # Sort memories by timestamp for context
        sorted_memories = sorted(
            self.memory_engine.memories, key=lambda m: m.timestamp, reverse=True
        )

        memories_to_cleanup = []
        memories_to_keep = []

        # Apply cleanup strategies
        for i, memory in enumerate(sorted_memories):
            context = {"memory_index": i, "total_memories": len(sorted_memories)}

            should_cleanup = False
            total_priority = 0.0

            for strategy in self.cleanup_strategies:
                if strategy.should_cleanup(memory, context):
                    should_cleanup = True
                    total_priority += strategy.get_priority(memory)

            if should_cleanup:
                memory.relevance_score = total_priority  # Store cleanup priority
                memories_to_cleanup.append(memory)
            else:
                memories_to_keep.append(memory)

        # Archive memories before cleanup if requested
        archived_count = 0
        if archive_before_cleanup and memories_to_cleanup and not dry_run:
            try:
                archive_info = self.archiver.archive_memories(
                    memories_to_cleanup,
                    {
                        "cleanup_reason": "scheduled_cleanup",
                        "strategies": [s.name for s in self.cleanup_strategies],
                    },
                )
                archived_count = archive_info.memory_count
            except Exception as e:
                self.logger.error(
                    "Failed to archive memories before cleanup",
                    extra={"error": str(e)},
                    exc_info=True,
                )
                # Continue with cleanup even if archival fails

        # Update memory engine with remaining memories
        if not dry_run:
            self.memory_engine.memories = memories_to_keep

            # Save updated memories
            if self.memory_engine.persist_path:
                self.memory_engine.save_memories()

        # Calculate stats
        final_count = len(memories_to_keep)
        cleaned_count = len(memories_to_cleanup)
        duration = (datetime.now() - start_time).total_seconds() * 1000

        stats = CleanupStats(
            total_memories_before=initial_count,
            total_memories_after=final_count,
            memories_cleaned=cleaned_count,
            memories_archived=archived_count,
            bytes_freed=0,  # Would need to calculate actual memory usage
            duration_ms=duration,
        )

        log_memory_operation(
            "cleanup",
            memories_before=initial_count,
            memories_after=final_count,
            memories_cleaned=cleaned_count,
            memories_archived=archived_count,
        )

        self.logger.info(
            "Memory cleanup completed",
            extra={
                "memories_before": initial_count,
                "memories_after": final_count,
                "memories_cleaned": cleaned_count,
                "memories_archived": archived_count,
                "duration_ms": duration,
                "dry_run": dry_run,
            },
        )

        return stats

    def auto_cleanup(
        self,
        max_memories: int = 1000,
        max_age_days: int = 30,
        min_relevance: float = 0.1,
    ) -> CleanupStats:
        """Perform automatic cleanup with sensible defaults"""
        # Clear existing strategies
        self.cleanup_strategies.clear()

        # Add default strategies
        self.add_cleanup_strategy(SizeBasedCleanup(max_memories))
        self.add_cleanup_strategy(AgeBasedCleanup(max_age_days))
        self.add_cleanup_strategy(RelevanceBasedCleanup(min_relevance))
        self.add_cleanup_strategy(DuplicateCleanup())

        return self.cleanup_memories(archive_before_cleanup=True)

    def export_memories(
        self,
        output_path: str,
        format: str = "json",
        filter_func: Optional[Callable[[Memory], bool]] = None,
    ):
        """Export memories to various formats"""
        memories = self.memory_engine.memories

        if filter_func:
            memories = [m for m in memories if filter_func(m)]

        self.logger.info(
            "Exporting memories",
            extra={
                "output_path": output_path,
                "format": format,
                "memory_count": len(memories),
            },
        )

        if format.lower() == "json":
            with open(output_path, "w") as f:
                json.dump(
                    [memory.to_dict() for memory in memories], f, indent=2, default=str
                )

        elif format.lower() == "csv":
            import csv

            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["content", "timestamp", "relevance_score", "metadata"])
                for memory in memories:
                    writer.writerow(
                        [
                            memory.content,
                            memory.timestamp.isoformat(),
                            memory.relevance_score,
                            json.dumps(memory.metadata),
                        ]
                    )

        elif format.lower() == "txt":
            with open(output_path, "w") as f:
                for memory in memories:
                    f.write(f"[{memory.timestamp.isoformat()}] {memory.content}\n")
                    if memory.metadata:
                        f.write(f"  Metadata: {json.dumps(memory.metadata)}\n")
                    f.write("\n")

        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(
            "Memory export completed",
            extra={
                "output_path": output_path,
                "format": format,
                "memory_count": len(memories),
            },
        )


def create_default_memory_manager(memory_engine: MemoryEngine) -> MemoryManager:
    """Create a memory manager with sensible default strategies"""
    manager = MemoryManager(memory_engine)

    # Add default cleanup strategies
    manager.add_cleanup_strategy(
        SizeBasedCleanup(max_memories=5000)
    )  # Keep 5000 most recent
    manager.add_cleanup_strategy(
        AgeBasedCleanup(max_age_days=90)
    )  # Remove memories older than 90 days
    manager.add_cleanup_strategy(
        RelevanceBasedCleanup(min_relevance=0.05)
    )  # Remove very low relevance
    manager.add_cleanup_strategy(
        DuplicateCleanup(similarity_threshold=0.95)
    )  # Remove duplicates

    return manager

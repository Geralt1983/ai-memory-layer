"""Provider-agnostic embedding interfaces using Python Protocols."""

from typing import List, Protocol, Optional, Dict, Any


class EmbeddingProvider(Protocol):
    """Provider-agnostic embedding interface.
    
    Implementations must preserve input order and return one vector per input.
    This uses Python's Protocol for structural typing - more pythonic than ABC.
    """
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            List of embedding vectors (same order as input)
            Each vector is a list of floats
        """
        ...
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text (convenience method).
        
        Args:
            text: String to embed
            
        Returns:
            Embedding vector or None if failed
        """
        ...
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider.
        
        Returns:
            Embedding vector dimension
        """
        ...


class BatchEmbeddingProvider(Protocol):
    """Extended interface for providers that support batch operations."""
    
    async def embed_batch_async(
        self, 
        texts: List[str], 
        batch_size: int = 100
    ) -> List[List[float]]:
        """Asynchronously embed texts in batches.
        
        Args:
            texts: List of strings to embed
            batch_size: Number of texts per batch
            
        Returns:
            List of embedding vectors (same order as input)
        """
        ...


class ConfigurableEmbeddingProvider(Protocol):
    """Interface for providers with runtime configuration."""
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Update provider configuration.
        
        Args:
            config: Configuration dictionary
        """
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration.
        
        Returns:
            Configuration dictionary
        """
        ...
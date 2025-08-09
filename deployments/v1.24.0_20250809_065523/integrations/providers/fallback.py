"""Fallback embedding provider for high availability."""

import logging
from typing import List, Optional, Dict, Any
from ..embeddings_interfaces import EmbeddingProvider

logger = logging.getLogger(__name__)


class FallbackEmbeddings:
    """Try primary provider first; on failure, fall back to backup.
    
    This provides high availability by automatically switching to a backup
    provider when the primary fails. Useful for production environments
    where uptime is critical.
    """
    
    def __init__(
        self, 
        primary: EmbeddingProvider, 
        backup: EmbeddingProvider,
        log_failures: bool = True
    ):
        """Initialize fallback provider.
        
        Args:
            primary: Primary embedding provider
            backup: Backup embedding provider  
            log_failures: Whether to log primary failures
        """
        self.primary = primary
        self.backup = backup
        self.log_failures = log_failures
        self._primary_failures = 0
        self._backup_uses = 0
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with automatic fallback.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            result = self.primary.embed(texts)
            return result
        except Exception as e:
            self._primary_failures += 1
            self._backup_uses += 1
            
            if self.log_failures:
                logger.warning(
                    f"Primary embedding provider failed (total failures: {self._primary_failures}), "
                    f"falling back to backup. Error: {e}"
                )
            
            try:
                return self.backup.embed(texts)
            except Exception as backup_error:
                logger.error(
                    f"Both primary and backup embedding providers failed. "
                    f"Primary error: {e}, Backup error: {backup_error}"
                )
                raise
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for single text with fallback.
        
        Args:
            text: String to embed
            
        Returns:
            Embedding vector or None if both fail
        """
        try:
            result = self.primary.embed_text(text)
            return result
        except Exception as e:
            self._primary_failures += 1
            self._backup_uses += 1
            
            if self.log_failures:
                logger.warning(
                    f"Primary embedding provider failed for single text, "
                    f"falling back to backup. Error: {e}"
                )
            
            try:
                return self.backup.embed_text(text)
            except Exception as backup_error:
                logger.error(
                    f"Both providers failed for single text. "
                    f"Primary: {e}, Backup: {backup_error}"
                )
                return None
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension from primary provider.
        
        Returns:
            Embedding dimension
        """
        try:
            return self.primary.get_embedding_dimension()
        except:
            return self.backup.get_embedding_dimension()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fallback statistics.
        
        Returns:
            Dictionary with failure and usage stats
        """
        return {
            "primary_failures": self._primary_failures,
            "backup_uses": self._backup_uses,
            "primary_provider": self.primary.__class__.__name__,
            "backup_provider": self.backup.__class__.__name__,
        }
    
    def reset_stats(self) -> None:
        """Reset failure and usage statistics."""
        self._primary_failures = 0
        self._backup_uses = 0


class MultiProviderFallback:
    """Advanced fallback with multiple providers in priority order."""
    
    def __init__(self, providers: List[EmbeddingProvider], log_failures: bool = True):
        """Initialize multi-provider fallback.
        
        Args:
            providers: List of providers in priority order
            log_failures: Whether to log failures
        """
        if not providers:
            raise ValueError("At least one provider required")
        
        self.providers = providers
        self.log_failures = log_failures
        self._failure_counts = [0] * len(providers)
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Try each provider in order until one succeeds.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If all providers fail
        """
        errors = []
        
        for i, provider in enumerate(self.providers):
            try:
                result = provider.embed(texts)
                if i > 0 and self.log_failures:
                    logger.info(f"Successfully used fallback provider #{i}")
                return result
            except Exception as e:
                self._failure_counts[i] += 1
                errors.append((provider.__class__.__name__, str(e)))
                
                if self.log_failures:
                    logger.warning(
                        f"Provider {i} ({provider.__class__.__name__}) failed: {e}"
                    )
        
        # All providers failed
        error_msg = "All embedding providers failed:\n"
        for provider_name, error in errors:
            error_msg += f"  - {provider_name}: {error}\n"
        
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Try each provider for single text embedding.
        
        Args:
            text: String to embed
            
        Returns:
            Embedding vector or None if all fail
        """
        for i, provider in enumerate(self.providers):
            try:
                result = provider.embed_text(text)
                if result is not None:
                    return result
            except Exception as e:
                self._failure_counts[i] += 1
                if self.log_failures:
                    logger.warning(f"Provider {i} failed for single text: {e}")
        
        return None
    
    def get_embedding_dimension(self) -> int:
        """Get dimension from first working provider.
        
        Returns:
            Embedding dimension
        """
        for provider in self.providers:
            try:
                return provider.get_embedding_dimension()
            except:
                continue
        
        # Default if all fail
        return 1536
    
    def get_stats(self) -> Dict[str, Any]:
        """Get multi-provider statistics.
        
        Returns:
            Dictionary with provider stats
        """
        return {
            "providers": [p.__class__.__name__ for p in self.providers],
            "failure_counts": self._failure_counts,
            "total_providers": len(self.providers),
        }
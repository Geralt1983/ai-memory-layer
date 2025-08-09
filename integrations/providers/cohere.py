"""Cohere embedding provider with lazy import and full functionality."""

import os
import logging
from typing import List, Optional, Literal
from ..embeddings_interfaces import EmbeddingProvider
from .provider_base import ProviderMixin


class CohereEmbeddings(ProviderMixin, EmbeddingProvider):
    """Cohere embeddings provider.
    
    Cohere offers multilingual embedding models with different sizes:
    - embed-english-v3.0: English embeddings (1024 dims)
    - embed-multilingual-v3.0: 100+ languages (1024 dims) 
    - embed-english-light-v3.0: Smaller English model (384 dims)
    - embed-multilingual-light-v3.0: Smaller multilingual (384 dims)
    
    To use:
    1. Install: pip install cohere
    2. Set COHERE_API_KEY environment variable
    3. Set EMBEDDING_PROVIDER=cohere
    """
    
    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: Optional[str] = None,
        input_type: Literal["search_document", "search_query", "classification", "clustering"] = "search_document",
        truncate: Literal["NONE", "START", "END"] = "END"
    ):
        """Initialize Cohere embeddings with lazy import.
        
        Args:
            model: Model name (embed-english-v3.0, embed-multilingual-v3.0, etc.)
            api_key: Cohere API key (or set COHERE_API_KEY env var)
            input_type: Type of input for optimized embeddings
            truncate: How to handle inputs longer than max length
        """
        self.model = model
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.input_type = input_type
        self.truncate = truncate
        # self._client will be set by ProviderMixin or _initialize_client
        self._available = False
        
        # Lazy initialization - will be called on first use
        self._initialize_client()
        
        # Dimension mapping for different models
        self.dimension_map = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384,
            # Legacy models
            "embed-english-v2.0": 4096,
            "embed-multilingual-v2.0": 768,
        }
    
    def _initialize_client(self):
        """Initialize Cohere client with lazy import."""
        # Allow test injection first
        if self._client is not None:
            self._available = True
            return
            
        try:
            import cohere
            if not self.api_key:
                raise ValueError("COHERE_API_KEY environment variable required")
            self._client = cohere.Client(api_key=self.api_key)
            self._available = True
            logging.info(f"Cohere client initialized with model {self.model}")
        except ImportError:
            logging.warning("Cohere package not installed. Run: pip install cohere")
            self._client = None
            self._available = False
        except Exception as e:
            logging.error(f"Failed to initialize Cohere client: {e}")
            self._client = None
            self._available = False
    
    def _sdk_available(self) -> bool:
        try:
            import importlib
            return importlib.util.find_spec("cohere") is not None
        except Exception:
            return False
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Cohere.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            List of embedding vectors
        """
        if self._client is None:
            self._initialize_client()
            if not self._available:
                raise RuntimeError(
                    "Cohere client not available. "
                    "Install cohere package and set COHERE_API_KEY."
                )
        
        if not texts:
            return []
            
        try:
            response = self._client.embed(
                texts=texts,
                model=self.model,
                input_type=self.input_type,
                truncate=self.truncate
            )
            return response.embeddings
        except Exception as e:
            logging.error(f"Cohere embedding failed: {e}")
            raise
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for single text.
        
        Args:
            text: String to embed
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            result = self.embed([text])
            return result[0] if result else None
        except Exception:
            return None
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension for current model.
        
        Returns:
            Embedding dimension
        """
        return self.dimension_map.get(self.model, 1024)
    
    def is_available(self) -> bool:
        """Check if Cohere embeddings are available."""
        return self._available
    
    def embed_for_search(
        self,
        documents: Optional[List[str]] = None,
        queries: Optional[List[str]] = None
    ) -> dict:
        """Optimized embeddings for search use case.
        
        Cohere recommends using different input_types for documents vs queries.
        
        Args:
            documents: List of documents to embed
            queries: List of queries to embed
            
        Returns:
            Dictionary with 'document_embeddings' and/or 'query_embeddings'
        """
        result = {}
        
        if documents:
            # Create a temporary instance for documents
            doc_embedder = CohereEmbeddings(
                model=self.model,
                api_key=self.api_key,
                input_type="search_document",
                truncate=self.truncate
            )
            result["document_embeddings"] = doc_embedder.embed(documents)
        
        if queries:
            # Create a temporary instance for queries  
            query_embedder = CohereEmbeddings(
                model=self.model,
                api_key=self.api_key,
                input_type="search_query",
                truncate=self.truncate
            )
            result["query_embeddings"] = query_embedder.embed(queries)
        
        return result
    
    def embed_with_compression(
        self,
        texts: List[str],
        compression_level: int = 0
    ) -> List[List[float]]:
        """Generate compressed embeddings (when supported).
        
        Some Cohere models support int8 compression for smaller storage.
        
        Args:
            texts: List of strings to embed
            compression_level: 0=none, 1=int8 (reduces size by 4x)
            
        Returns:
            List of embedding vectors (possibly compressed)
        """
        # TODO: Implement compression when available
        return self.embed(texts)
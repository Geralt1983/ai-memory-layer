"""
Transformer-based Embedding Provider using BERT for semantic understanding
"""

import numpy as np
from typing import List, Union, Optional
from core.logging_config import get_logger, monitor_performance
from integrations.embeddings import EmbeddingProvider


try:
    import torch
    from transformers import BertTokenizer, BertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers library not available. Install with: pip install transformers torch")
    
    # Create mock torch module for compatibility
    class MockTorch:
        @staticmethod
        def device(device_name):
            return device_name
        
        @staticmethod
        def cuda():
            return MockTorch()
        
        @staticmethod
        def is_available():
            return False
        
        @staticmethod
        def no_grad():
            return MockContextManager()
    
    class MockContextManager:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    torch = MockTorch()


class TransformerEmbeddingProvider(EmbeddingProvider):
    """
    BERT-based embedding provider for semantic understanding
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', device: Optional[str] = None):
        """
        Initialize the transformer embedding provider
        
        Args:
            model_name: HuggingFace model name (default: bert-base-uncased)
            device: Device to use ('cpu', 'cuda', or None for auto-detection)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required. Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.logger = get_logger("transformer_embeddings")
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Loading transformer model: {model_name} on {self.device}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.logger.info(f"Successfully loaded {model_name}")
            
            # Get embedding dimension
            self.embedding_dim = self.model.config.hidden_size
            
        except Exception as e:
            self.logger.error(f"Failed to load transformer model: {e}")
            raise
    
    @monitor_performance("transformer_embed_text")
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate semantic embedding using BERT
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(text, list):
            text = " ".join(text)
        
        if not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.embedding_dim)
        
        text_length = len(text)
        self.logger.debug(f"Generating transformer embedding for text of length {text_length}")
        
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512  # BERT's max sequence length
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            # Convert to numpy and ensure it's on CPU
            embedding_array = embeddings.cpu().numpy()
            
            # Handle single token case (ensure 1D array)
            if embedding_array.ndim == 0:
                embedding_array = np.array([embedding_array])
            
            self.logger.debug(f"Generated embedding of shape {embedding_array.shape}")
            return embedding_array
            
        except Exception as e:
            self.logger.error(f"Failed to generate transformer embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim)
    
    @monitor_performance("transformer_embed_batch")
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of numpy arrays (embeddings)
        """
        if not texts:
            return []
        
        batch_size = len(texts)
        self.logger.debug(f"Generating batch embeddings for {batch_size} texts")
        
        try:
            # Tokenize all texts in batch
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden state for each text
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Convert to list of numpy arrays
            embedding_list = [emb.cpu().numpy() for emb in embeddings]
            
            self.logger.info(f"Generated {len(embedding_list)} batch embeddings")
            return embedding_list
            
        except Exception as e:
            self.logger.error(f"Failed to generate batch transformer embeddings: {e}")
            # Return zero vectors as fallback
            return [np.zeros(self.embedding_dim) for _ in texts]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider"""
        return self.embedding_dim
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'embedding_dimension': self.embedding_dim,
            'max_sequence_length': 512,
            'type': 'transformer_bert'
        }


class MockTransformerEmbeddingProvider(EmbeddingProvider):
    """
    Mock transformer embedding provider for environments where transformers are not available
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.logger = get_logger("mock_transformer_embeddings")
        self.logger.warning("Using mock transformer embeddings - install transformers for real BERT embeddings")
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate mock embedding (random vector)"""
        if isinstance(text, list):
            text = " ".join(text)
        
        # Generate deterministic "embedding" based on text hash
        import hashlib
        text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(text_hash % 2**32)
        return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate mock batch embeddings"""
        return [self.embed_text(text) for text in texts]
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim
    
    def get_model_info(self) -> dict:
        return {
            'model_name': 'mock_bert',
            'device': 'cpu',
            'embedding_dimension': self.embedding_dim,
            'max_sequence_length': 512,
            'type': 'mock_transformer'
        }


def create_transformer_embedding_provider(
    model_name: str = 'bert-base-uncased',
    device: Optional[str] = None,
    fallback_to_mock: bool = True
) -> EmbeddingProvider:
    """
    Factory function to create transformer embedding provider with fallback
    
    Args:
        model_name: HuggingFace model name
        device: Device to use
        fallback_to_mock: Whether to use mock provider if transformers unavailable
        
    Returns:
        EmbeddingProvider instance
    """
    try:
        if TRANSFORMERS_AVAILABLE:
            return TransformerEmbeddingProvider(model_name=model_name, device=device)
        elif fallback_to_mock:
            return MockTransformerEmbeddingProvider()
        else:
            raise ImportError("Transformers library not available")
    except Exception as e:
        logger = get_logger("transformer_factory")
        logger.error(f"Failed to create transformer embedding provider: {e}")
        if fallback_to_mock:
            logger.warning("Falling back to mock transformer embeddings")
            return MockTransformerEmbeddingProvider()
        else:
            raise
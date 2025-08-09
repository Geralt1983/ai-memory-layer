"""
Transformer-enhanced MemoryEngine factory and utilities
"""

from typing import Optional
from core.memory_engine import MemoryEngine
from integrations.transformer_embeddings import create_transformer_embedding_provider
from core.logging_config import get_logger

def create_transformer_memory_engine(
    model_name: str = 'bert-base-uncased',
    device: Optional[str] = None,
    fallback_to_mock: bool = True,
    vector_store=None,
    persist_path: Optional[str] = None
) -> MemoryEngine:
    """
    Create a MemoryEngine with transformer-based embeddings (BERT)
    
    Args:
        model_name: HuggingFace model name for BERT
        device: Device to run model on ('cpu', 'cuda', or None for auto)
        fallback_to_mock: Whether to use mock embeddings if transformers unavailable
        vector_store: Vector store to use (if None, will be created automatically)
        persist_path: Path for persistent memory storage
        
    Returns:
        MemoryEngine instance with transformer embeddings
    """
    logger = get_logger("transformer_memory_engine")
    
    try:
        # Create transformer embedding provider
        embedding_provider = create_transformer_embedding_provider(
            model_name=model_name,
            device=device,
            fallback_to_mock=fallback_to_mock
        )
        
        # Create vector store if not provided
        if vector_store is None:
            try:
                from storage.faiss_store import FaissVectorStore
                vector_store = FaissVectorStore(
                    embedding_dim=embedding_provider.get_embedding_dimension()
                )
                logger.info(f"Created FAISS vector store with {embedding_provider.get_embedding_dimension()}D embeddings")
            except ImportError:
                logger.warning("FAISS not available, creating memory engine without vector store")
                vector_store = None
        
        # Create memory engine
        memory_engine = MemoryEngine(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            persist_path=persist_path
        )
        
        model_info = embedding_provider.get_model_info()
        logger.info(
            "Created transformer-enhanced MemoryEngine",
            extra={
                "model_name": model_info.get('model_name'),
                "embedding_dimension": model_info.get('embedding_dimension'),
                "device": model_info.get('device'),
                "vector_store_type": type(vector_store).__name__ if vector_store else None
            }
        )
        
        return memory_engine
        
    except Exception as e:
        logger.error(f"Failed to create transformer memory engine: {e}")
        # Fallback to regular memory engine with OpenAI embeddings
        logger.info("Falling back to OpenAI embeddings")
        
        try:
            from integrations.openai_embeddings import OpenAIEmbeddings
            embedding_provider = OpenAIEmbeddings()
            
            if vector_store is None:
                try:
                    from storage.faiss_store import FaissVectorStore
                    vector_store = FaissVectorStore(embedding_dim=1536)  # OpenAI ada-002 dimension
                except ImportError:
                    vector_store = None
            
            return MemoryEngine(
                vector_store=vector_store,
                embedding_provider=embedding_provider,
                persist_path=persist_path
            )
        except Exception as fallback_error:
            logger.error(f"Fallback to OpenAI also failed: {fallback_error}")
            raise


def upgrade_memory_engine_to_transformer(
    existing_engine: MemoryEngine,
    model_name: str = 'bert-base-uncased',
    device: Optional[str] = None,
    regenerate_embeddings: bool = False
) -> MemoryEngine:
    """
    Upgrade an existing MemoryEngine to use transformer embeddings
    
    Args:
        existing_engine: Existing MemoryEngine to upgrade
        model_name: BERT model to use
        device: Device for the model
        regenerate_embeddings: Whether to regenerate embeddings for existing memories
        
    Returns:
        Upgraded MemoryEngine with transformer embeddings
    """
    logger = get_logger("transformer_memory_upgrade")
    
    try:
        # Create new transformer embedding provider
        transformer_provider = create_transformer_embedding_provider(
            model_name=model_name,
            device=device,
            fallback_to_mock=True
        )
        
        # Create new memory engine with same config but new embeddings
        upgraded_engine = MemoryEngine(
            vector_store=existing_engine.vector_store,
            embedding_provider=transformer_provider,
            persist_path=existing_engine.persist_path
        )
        
        # Copy existing memories
        upgraded_engine.memories = existing_engine.memories.copy()
        
        if regenerate_embeddings:
            logger.info(f"Regenerating embeddings for {len(upgraded_engine.memories)} memories...")
            
            for i, memory in enumerate(upgraded_engine.memories):
                try:
                    # Generate new embedding with transformer model
                    new_embedding = transformer_provider.embed_text(memory.content)
                    memory.embedding = new_embedding
                    
                    # Update in vector store if available
                    if upgraded_engine.vector_store and new_embedding is not None:
                        # Note: This assumes vector store supports updating embeddings
                        # Some implementations might need to delete and re-add
                        pass
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"Regenerated embeddings for {i + 1}/{len(upgraded_engine.memories)} memories")
                        
                except Exception as e:
                    logger.warning(f"Failed to regenerate embedding for memory {i}: {e}")
            
            logger.info("Completed embedding regeneration")
        
        model_info = transformer_provider.get_model_info()
        logger.info(
            "Successfully upgraded MemoryEngine to transformer embeddings",
            extra={
                "model_name": model_info.get('model_name'),
                "embedding_dimension": model_info.get('embedding_dimension'),
                "device": model_info.get('device'),
                "memories_count": len(upgraded_engine.memories),
                "regenerated_embeddings": regenerate_embeddings
            }
        )
        
        return upgraded_engine
        
    except Exception as e:
        logger.error(f"Failed to upgrade MemoryEngine: {e}")
        logger.info("Returning original MemoryEngine unchanged")
        return existing_engine
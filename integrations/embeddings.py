from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import openai
from openai import OpenAI
from ..core.logging_config import get_logger, monitor_performance, log_embedding_operation


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        pass


class OpenAIEmbeddings(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.logger = get_logger("embeddings")
        
        self.logger.info("OpenAI embeddings initialized", extra={"model": model})
    
    @monitor_performance("embed_text")
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, list):
            text = " ".join(text)
        
        text_length = len(text)
        self.logger.debug("Generating embedding", extra={
            "text_length": text_length,
            "model": self.model
        })
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            embedding = np.array(response.data[0].embedding)
            
            log_embedding_operation("embed_text", text_length, self.model, 
                                   embedding_dimension=len(embedding))
            
            self.logger.debug("Embedding generated successfully", extra={
                "text_length": text_length,
                "embedding_dimension": len(embedding)
            })
            
            return embedding
            
        except Exception as e:
            self.logger.error("Failed to generate embedding", extra={
                "error": str(e),
                "text_length": text_length,
                "model": self.model
            }, exc_info=True)
            raise
    
    @monitor_performance("embed_batch")
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        total_length = sum(len(text) for text in texts)
        self.logger.debug("Generating batch embeddings", extra={
            "batch_size": len(texts),
            "total_text_length": total_length,
            "model": self.model
        })
        
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            
            embeddings = [np.array(item.embedding) for item in response.data]
            
            log_embedding_operation("embed_batch", total_length, self.model,
                                   batch_size=len(texts),
                                   embedding_dimension=len(embeddings[0]) if embeddings else 0)
            
            self.logger.info("Batch embeddings generated successfully", extra={
                "batch_size": len(texts),
                "total_text_length": total_length,
                "embedding_dimension": len(embeddings[0]) if embeddings else 0
            })
            
            return embeddings
            
        except Exception as e:
            self.logger.error("Failed to generate batch embeddings", extra={
                "error": str(e),
                "batch_size": len(texts),
                "total_text_length": total_length,
                "model": self.model
            }, exc_info=True)
            raise
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import openai
from openai import OpenAI


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
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, list):
            text = " ".join(text)
        
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        
        return np.array(response.data[0].embedding)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        
        return [np.array(item.embedding) for item in response.data]
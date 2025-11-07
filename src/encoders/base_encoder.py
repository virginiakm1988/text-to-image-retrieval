"""
Base encoder abstract class
"""
from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import List, Union
from PIL import Image


class BaseEncoder(ABC):
    """Base encoder class that defines common interface for image and text encoding"""
    
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        
    @abstractmethod
    def load_model(self):
        """Load pretrained model"""
        pass
    
    @abstractmethod
    def encode_images(self, images: List[Union[str, Image.Image]]) -> np.ndarray:
        """
        Encode images to vectors
        
        Args:
            images: List of image paths or PIL Image objects
            
        Returns:
            numpy array with shape (n_images, embedding_dim)
        """
        pass
    
    @abstractmethod
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to vectors
        
        Args:
            texts: List of texts
            
        Returns:
            numpy array with shape (n_texts, embedding_dim)
        """
        pass
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embedding vectors"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          image_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query vector and image vectors"""
        # Ensure vectors are normalized
        query_norm = self.normalize_embeddings(query_embedding.reshape(1, -1))
        image_norm = self.normalize_embeddings(image_embeddings)
        
        # Compute cosine similarity
        similarities = np.dot(query_norm, image_norm.T).flatten()
        return similarities

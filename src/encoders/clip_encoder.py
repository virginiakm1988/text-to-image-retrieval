"""
CLIP编码器实现
"""
import torch
import numpy as np
from typing import List, Union
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from .base_encoder import BaseEncoder


class CLIPEncoder(BaseEncoder):
    """基于CLIP的图像和文本编码器"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        super().__init__(model_name, device)
        self.load_model()
    
    def load_model(self):
        """加载CLIP预训练模型"""
        print(f"Loading CLIP model: {self.model_name}")
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on device: {self.device}")
    
    def encode_images(self, images: List[Union[str, Image.Image]], batch_size: int = 32) -> np.ndarray:
        """
        编码图像为向量
        
        Args:
            images: 图像路径列表或PIL Image对象列表
            batch_size: 批处理大小
            
        Returns:
            numpy数组，形状为(n_images, embedding_dim)
        """
        all_embeddings = []
        
        # 预处理图像
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            elif isinstance(img, Image.Image):
                img = img.convert('RGB')
            processed_images.append(img)
        
        # 分批处理
        for i in range(0, len(processed_images), batch_size):
            batch_images = processed_images[i:i + batch_size]
            
            with torch.no_grad():
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                all_embeddings.append(image_features.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            numpy数组，形状为(n_texts, embedding_dim)
        """
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            with torch.no_grad():
                inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                all_embeddings.append(text_features.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def get_embedding_dim(self) -> int:
        """获取嵌入向量维度"""
        return self.model.config.projection_dim


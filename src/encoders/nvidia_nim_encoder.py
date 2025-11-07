"""
NVIDIA NIM编码器实现
支持NVIDIA的视觉-语言模型和嵌入模型
"""
import base64
import io
import numpy as np
from typing import List, Union, Optional
from PIL import Image
from openai import OpenAI
from .base_encoder import BaseEncoder


class NVIDIANIMEncoder(BaseEncoder):
    """基于NVIDIA NIM的图像和文本编码器"""
    
    def __init__(self, 
                 model_name: str = "nvidia/nvclip",
                 api_key: Optional[str] = None,
                 base_url: str = "https://integrate.api.nvidia.com/v1",
                 device: str = None):
        """
        初始化NVIDIA NIM编码器
        
        Args:
            model_name: 模型名称 (nvidia/nvclip, nvidia/nv-dinov2等)
            api_key: NVIDIA NIM API密钥
            base_url: API基础URL
            device: 设备（NIM是云端服务，此参数仅为兼容性保留）
        """
        super().__init__(model_name, device)
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        self.load_model()
    
    def load_model(self):
        """初始化NVIDIA NIM客户端"""
        print(f"Initializing NVIDIA NIM client for model: {self.model_name}")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        print(f"NVIDIA NIM client initialized")
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """将PIL图像转换为base64编码字符串"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
    
    def encode_images(self, images: List[Union[str, Image.Image]], batch_size: int = 10) -> np.ndarray:
        """
        使用NVIDIA NIM编码图像为向量
        
        Args:
            images: 图像路径列表或PIL Image对象列表
            batch_size: 批处理大小（NIM有请求限制）
            
        Returns:
            numpy数组，形状为(n_images, embedding_dim)
        """
        all_embeddings = []
        
        # 预处理图像为base64格式
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            elif isinstance(img, Image.Image):
                img = img.convert('RGB')
            
            img_base64 = self._image_to_base64(img)
            processed_images.append(img_base64)
        
        # 分批处理（NIM API通常有批量限制）
        for i in range(0, len(processed_images), batch_size):
            batch_images = processed_images[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch_images,
                    model=self.model_name,
                    encoding_format="float"
                )
                
                # 提取嵌入向量
                batch_embeddings = []
                for embedding_obj in response.data:
                    batch_embeddings.append(embedding_obj.embedding)
                
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error encoding batch {i//batch_size + 1}: {e}")
                # 为失败的批次添加零向量
                dummy_embedding = [0.0] * self.get_embedding_dim()
                all_embeddings.extend([dummy_embedding] * len(batch_images))
        
        embeddings_array = np.array(all_embeddings)
        
        # 归一化嵌入向量
        return self.normalize_embeddings(embeddings_array)
    
    def encode_text(self, texts: List[str], batch_size: int = 20) -> np.ndarray:
        """
        使用NVIDIA NIM编码文本为向量
        
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
            
            try:
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.model_name,
                    encoding_format="float"
                )
                
                # 提取嵌入向量
                batch_embeddings = []
                for embedding_obj in response.data:
                    batch_embeddings.append(embedding_obj.embedding)
                
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error encoding text batch {i//batch_size + 1}: {e}")
                # 为失败的批次添加零向量
                dummy_embedding = [0.0] * self.get_embedding_dim()
                all_embeddings.extend([dummy_embedding] * len(batch_texts))
        
        embeddings_array = np.array(all_embeddings)
        
        # 归一化嵌入向量
        return self.normalize_embeddings(embeddings_array)
    
    def encode_mixed_input(self, inputs: List[Union[str, Image.Image]], batch_size: int = 10) -> np.ndarray:
        """
        编码混合输入（文本和图像）
        
        Args:
            inputs: 混合输入列表，可以包含文本字符串、图像路径或PIL Image对象
            batch_size: 批处理大小
            
        Returns:
            numpy数组，形状为(n_inputs, embedding_dim)
        """
        all_embeddings = []
        
        # 预处理输入
        processed_inputs = []
        for inp in inputs:
            if isinstance(inp, str):
                # 检查是否是图像路径
                if inp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                    # 图像路径
                    img = Image.open(inp).convert('RGB')
                    img_base64 = self._image_to_base64(img)
                    processed_inputs.append(img_base64)
                else:
                    # 文本
                    processed_inputs.append(inp)
            elif isinstance(inp, Image.Image):
                # PIL图像
                img_base64 = self._image_to_base64(inp.convert('RGB'))
                processed_inputs.append(img_base64)
            else:
                # 其他类型转为字符串
                processed_inputs.append(str(inp))
        
        # 分批处理
        for i in range(0, len(processed_inputs), batch_size):
            batch_inputs = processed_inputs[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch_inputs,
                    model=self.model_name,
                    encoding_format="float"
                )
                
                # 提取嵌入向量
                batch_embeddings = []
                for embedding_obj in response.data:
                    batch_embeddings.append(embedding_obj.embedding)
                
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error encoding mixed batch {i//batch_size + 1}: {e}")
                # 为失败的批次添加零向量
                dummy_embedding = [0.0] * self.get_embedding_dim()
                all_embeddings.extend([dummy_embedding] * len(batch_inputs))
        
        embeddings_array = np.array(all_embeddings)
        
        # 归一化嵌入向量
        return self.normalize_embeddings(embeddings_array)
    
    def get_embedding_dim(self) -> int:
        """
        获取嵌入向量维度
        
        不同的NVIDIA NIM模型有不同的维度：
        - nvidia/nvclip: 512
        - nvidia/nv-dinov2: 1024 (根据模型变体)
        """
        if "nvclip" in self.model_name.lower():
            return 512
        elif "dinov2" in self.model_name.lower():
            return 1024  # 可能需要根据具体模型调整
        else:
            # 默认维度，可能需要根据实际模型调整
            return 512
    
    def get_available_models(self) -> List[str]:
        """获取可用的NVIDIA NIM模型列表"""
        return [
            "nvidia/nvclip",
            "nvidia/nv-dinov2",
            "nvidia/vila",
            "meta/llama-3.2-90b-vision-instruct",
            "meta/llama-3.2-11b-vision-instruct",
            "nvidia/cosmos-reason1-7b",
            "google/paligemma"
        ]


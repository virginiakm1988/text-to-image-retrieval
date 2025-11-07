"""
图像检索系统核心实现
"""
import os
import glob
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from PIL import Image
import json
from tqdm import tqdm

from ..encoders import CLIPEncoder, SigLIPEncoder, NVIDIANIMEncoder
from ..indexing import FAISSIndex


class ImageRetrievalSystem:
    """以文搜图检索系统"""
    
    def __init__(self, 
                 encoder_type: str = "clip",
                 model_name: Optional[str] = None,
                 index_type: str = "flat",
                 device: str = None,
                 nvidia_api_key: Optional[str] = None):
        """
        初始化图像检索系统
        
        Args:
            encoder_type: 编码器类型 ("clip", "siglip", "nvidia_nim")
            model_name: 模型名称，如果为None则使用默认模型
            index_type: FAISS索引类型 ("flat", "ivf", "hnsw")
            device: 计算设备
            nvidia_api_key: NVIDIA NIM API密钥（仅当encoder_type="nvidia_nim"时需要）
        """
        self.encoder_type = encoder_type
        self.index_type = index_type
        self.device = device
        self.nvidia_api_key = nvidia_api_key
        
        # 初始化编码器
        self.encoder = self._create_encoder(encoder_type, model_name)
        
        # 初始化索引
        embedding_dim = self.encoder.get_embedding_dim()
        self.index = FAISSIndex(embedding_dim, index_type)
        
        # 存储图像信息
        self.image_database = {}  # {image_path: metadata}
        
        print(f"Initialized ImageRetrievalSystem with {encoder_type} encoder and {index_type} index")
    
    def _create_encoder(self, encoder_type: str, model_name: Optional[str]):
        """创建编码器实例"""
        if encoder_type.lower() == "clip":
            model_name = model_name or "openai/clip-vit-base-patch32"
            return CLIPEncoder(model_name, self.device)
        elif encoder_type.lower() == "siglip":
            model_name = model_name or "google/siglip-base-patch16-224"
            return SigLIPEncoder(model_name, self.device)
        elif encoder_type.lower() == "nvidia_nim":
            model_name = model_name or "nvidia/nvclip"
            return NVIDIANIMEncoder(model_name, self.nvidia_api_key, device=self.device)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
    
    def add_images_from_directory(self, 
                                image_dir: str, 
                                supported_formats: List[str] = None,
                                batch_size: int = 32,
                                recursive: bool = True) -> int:
        """
        从目录添加图像到索引
        
        Args:
            image_dir: 图像目录路径
            supported_formats: 支持的图像格式列表
            batch_size: 批处理大小
            recursive: 是否递归搜索子目录
            
        Returns:
            添加的图像数量
        """
        if supported_formats is None:
            supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        
        # 收集图像文件
        image_paths = []
        for fmt in supported_formats:
            if recursive:
                pattern = os.path.join(image_dir, '**', fmt)
                image_paths.extend(glob.glob(pattern, recursive=True))
            else:
                pattern = os.path.join(image_dir, fmt)
                image_paths.extend(glob.glob(pattern))
        
        if not image_paths:
            print(f"No images found in {image_dir}")
            return 0
        
        print(f"Found {len(image_paths)} images. Starting encoding...")
        
        # 分批处理图像
        added_count = 0
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i:i + batch_size]
            
            try:
                # 编码图像
                embeddings = self.encoder.encode_images(batch_paths, batch_size)
                
                # 创建元数据
                metadata_list = []
                for path in batch_paths:
                    metadata = self._extract_image_metadata(path)
                    self.image_database[path] = metadata
                    metadata_list.append(metadata)
                
                # 添加到索引
                self.index.add_vectors(embeddings, batch_paths, metadata_list)
                added_count += len(batch_paths)
                
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")
                continue
        
        print(f"Successfully added {added_count} images to the index")
        return added_count
    
    def _extract_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """提取图像元数据"""
        try:
            with Image.open(image_path) as img:
                metadata = {
                    'filename': os.path.basename(image_path),
                    'path': image_path,
                    'size': img.size,
                    'format': img.format,
                    'mode': img.mode,
                    'file_size': os.path.getsize(image_path)
                }
                
                # 添加EXIF信息（如果有）
                if hasattr(img, '_getexif') and img._getexif():
                    metadata['exif'] = dict(img._getexif())
                
                return metadata
        except Exception as e:
            return {
                'filename': os.path.basename(image_path),
                'path': image_path,
                'error': str(e)
            }
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               return_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        使用文本查询搜索相似图像
        
        Args:
            query: 查询文本
            top_k: 返回最相似的k个结果
            return_metadata: 是否返回图像元数据
            
        Returns:
            搜索结果列表，每个结果包含图像路径、相似度分数和元数据
        """
        if self.index.index.ntotal == 0:
            print("Index is empty. Please add images first.")
            return []
        
        # 编码查询文本
        query_embedding = self.encoder.encode_text([query])
        
        # 搜索相似向量
        scores, indices = self.index.search(query_embedding, top_k)
        
        # 构建结果
        results = []
        for score, idx in zip(scores, indices):
            image_path = self.index.get_image_path(idx)
            
            result = {
                'image_path': image_path,
                'similarity_score': float(score),
                'rank': len(results) + 1
            }
            
            if return_metadata:
                result['metadata'] = self.image_database.get(image_path, {})
            
            results.append(result)
        
        return results
    
    def search_by_image(self, 
                       image: Union[str, Image.Image], 
                       top_k: int = 10,
                       return_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        使用图像搜索相似图像
        
        Args:
            image: 查询图像（路径或PIL Image对象）
            top_k: 返回最相似的k个结果
            return_metadata: 是否返回图像元数据
            
        Returns:
            搜索结果列表
        """
        if self.index.index.ntotal == 0:
            print("Index is empty. Please add images first.")
            return []
        
        # 编码查询图像
        query_embedding = self.encoder.encode_images([image])
        
        # 搜索相似向量
        scores, indices = self.index.search(query_embedding, top_k)
        
        # 构建结果
        results = []
        for score, idx in zip(scores, indices):
            image_path = self.index.get_image_path(idx)
            
            result = {
                'image_path': image_path,
                'similarity_score': float(score),
                'rank': len(results) + 1
            }
            
            if return_metadata:
                result['metadata'] = self.image_database.get(image_path, {})
            
            results.append(result)
        
        return results
    
    def save_system(self, save_path: str):
        """
        保存整个检索系统
        
        Args:
            save_path: 保存路径（不包含扩展名）
        """
        # 保存FAISS索引
        self.index.save_index(save_path)
        
        # 保存系统配置和图像数据库
        system_config = {
            'encoder_type': self.encoder_type,
            'model_name': self.encoder.model_name,
            'index_type': self.index_type,
            'image_database': self.image_database,
            'embedding_dim': self.encoder.get_embedding_dim()
        }
        
        with open(f"{save_path}_config.json", 'w', encoding='utf-8') as f:
            json.dump(system_config, f, ensure_ascii=False, indent=2)
        
        print(f"System saved to {save_path}")
    
    def load_system(self, load_path: str, nvidia_api_key: Optional[str] = None):
        """
        加载检索系统
        
        Args:
            load_path: 加载路径（不包含扩展名）
            nvidia_api_key: NVIDIA NIM API密钥（如果需要）
        """
        # 加载系统配置
        with open(f"{load_path}_config.json", 'r', encoding='utf-8') as f:
            system_config = json.load(f)
        
        # 重新初始化编码器
        self.encoder_type = system_config['encoder_type']
        self.nvidia_api_key = nvidia_api_key
        self.encoder = self._create_encoder(
            system_config['encoder_type'], 
            system_config['model_name']
        )
        
        # 加载FAISS索引
        self.index.load_index(load_path)
        
        # 加载图像数据库
        self.image_database = system_config['image_database']
        
        print(f"System loaded from {load_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        index_stats = self.index.get_stats()
        
        return {
            'encoder_type': self.encoder_type,
            'model_name': self.encoder.model_name,
            'total_images': len(self.image_database),
            'index_stats': index_stats,
            'embedding_dim': self.encoder.get_embedding_dim()
        }
    
    def get_random_images(self, count: int = 5) -> List[str]:
        """获取随机图像路径用于展示"""
        image_paths = list(self.image_database.keys())
        if len(image_paths) <= count:
            return image_paths
        
        import random
        return random.sample(image_paths, count)


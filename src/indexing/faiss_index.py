"""
FAISS向量索引实现
"""
import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional, Dict, Any


class FAISSIndex:
    """FAISS向量索引管理器"""
    
    def __init__(self, embedding_dim: int, index_type: str = "flat"):
        """
        初始化FAISS索引
        
        Args:
            embedding_dim: 向量维度
            index_type: 索引类型 ("flat", "ivf", "hnsw")
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.image_paths = []
        self.metadata = {}
        
        self._create_index()
    
    def _create_index(self):
        """创建FAISS索引"""
        if self.index_type == "flat":
            # 精确搜索，适合小数据集
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # 内积索引
        elif self.index_type == "ivf":
            # 倒排文件索引，适合大数据集
            nlist = 100  # 聚类中心数量
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        elif self.index_type == "hnsw":
            # 分层导航小世界图，适合高维数据
            M = 16  # 连接数
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, M)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        print(f"Created {self.index_type} index with dimension {self.embedding_dim}")
    
    def add_vectors(self, embeddings: np.ndarray, image_paths: List[str], 
                   metadata: Optional[List[Dict[str, Any]]] = None):
        """
        添加向量到索引
        
        Args:
            embeddings: 图像嵌入向量，形状为(n_images, embedding_dim)
            image_paths: 对应的图像路径列表
            metadata: 可选的元数据列表
        """
        # 确保向量已归一化（对于余弦相似度）
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # 如果是IVF索引，需要先训练
        if self.index_type == "ivf" and not self.index.is_trained:
            print("Training IVF index...")
            self.index.train(embeddings.astype(np.float32))
        
        # 添加向量到索引
        start_id = len(self.image_paths)
        self.index.add(embeddings.astype(np.float32))
        
        # 保存路径和元数据
        self.image_paths.extend(image_paths)
        if metadata:
            for i, meta in enumerate(metadata):
                self.metadata[start_id + i] = meta
        
        print(f"Added {len(embeddings)} vectors to index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> Tuple[List[float], List[int]]:
        """
        搜索最相似的向量
        
        Args:
            query_embedding: 查询向量，形状为(1, embedding_dim)
            top_k: 返回最相似的k个结果
            
        Returns:
            (相似度分数列表, 索引ID列表)
        """
        if self.index.ntotal == 0:
            return [], []
        
        # 归一化查询向量
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # 设置搜索参数（对于IVF索引）
        if self.index_type == "ivf":
            self.index.nprobe = min(10, self.index.nlist)  # 搜索的聚类数量
        
        # 执行搜索
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # 过滤有效结果
        valid_results = [(score, idx) for score, idx in zip(scores[0], indices[0]) if idx != -1]
        
        if valid_results:
            scores, indices = zip(*valid_results)
            return list(scores), list(indices)
        else:
            return [], []
    
    def get_image_path(self, index_id: int) -> str:
        """根据索引ID获取图像路径"""
        if 0 <= index_id < len(self.image_paths):
            return self.image_paths[index_id]
        return ""
    
    def get_metadata(self, index_id: int) -> Dict[str, Any]:
        """根据索引ID获取元数据"""
        return self.metadata.get(index_id, {})
    
    def save_index(self, index_path: str):
        """
        保存索引到文件
        
        Args:
            index_path: 索引文件路径（不包含扩展名）
        """
        # 保存FAISS索引
        faiss.write_index(self.index, f"{index_path}.faiss")
        
        # 保存元数据
        metadata_dict = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'image_paths': self.image_paths,
            'metadata': self.metadata
        }
        
        with open(f"{index_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata_dict, f)
        
        print(f"Index saved to {index_path}.faiss and {index_path}_metadata.pkl")
    
    def load_index(self, index_path: str):
        """
        从文件加载索引
        
        Args:
            index_path: 索引文件路径（不包含扩展名）
        """
        # 加载FAISS索引
        if os.path.exists(f"{index_path}.faiss"):
            self.index = faiss.read_index(f"{index_path}.faiss")
        else:
            raise FileNotFoundError(f"Index file not found: {index_path}.faiss")
        
        # 加载元数据
        if os.path.exists(f"{index_path}_metadata.pkl"):
            with open(f"{index_path}_metadata.pkl", 'rb') as f:
                metadata_dict = pickle.load(f)
            
            self.embedding_dim = metadata_dict['embedding_dim']
            self.index_type = metadata_dict['index_type']
            self.image_paths = metadata_dict['image_paths']
            self.metadata = metadata_dict['metadata']
        else:
            raise FileNotFoundError(f"Metadata file not found: {index_path}_metadata.pkl")
        
        print(f"Index loaded from {index_path}. Total vectors: {self.index.ntotal}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'total_images': len(self.image_paths)
        }


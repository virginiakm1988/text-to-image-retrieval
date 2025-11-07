# 以文搜图检索系统 (Text-to-Image Retrieval System)

基于CLIP/SigLIP + FAISS向量索引的图像检索系统，支持使用自然语言描述搜索相关图片。

## 功能特性

- 🔍 **以文搜图**: 使用自然语言描述搜索相关图片
- 🚀 **高效检索**: 基于FAISS向量索引，支持快速相似度搜索
- 🎯 **多模型支持**: 支持CLIP、SigLIP和NVIDIA NIM预训练模型
- 🌐 **Web界面**: 提供直观的Streamlit Web界面
- 📊 **可视化结果**: 展示检索结果和相似度分数
- ☁️ **云端推理**: 支持NVIDIA NIM云端视觉-语言模型服务

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 构建图像索引

```bash
python build_index.py --image_dir ./images --index_path ./image_index.faiss
```

### 2. 启动Web界面

```bash
streamlit run app.py
```

### 3. 使用API

```python
from image_retrieval import ImageRetrievalSystem

# 初始化系统
retrieval_system = ImageRetrievalSystem()
retrieval_system.load_index("./image_index.faiss")

# 搜索图片
results = retrieval_system.search("a cat sitting on a chair", top_k=5)
```

## 项目结构

```
image-retrieval/
├── src/
│   ├── encoders/          # 编码器模块
│   ├── indexing/          # 向量索引模块
│   └── retrieval/         # 检索系统核心
├── app.py                 # Streamlit Web应用
├── build_index.py         # 构建索引脚本
├── requirements.txt       # 项目依赖
└── README.md             # 项目说明
```

## 支持的模型

### 本地模型
- **CLIP**: OpenAI的经典视觉-语言模型
  - `openai/clip-vit-base-patch32` (默认)
  - `openai/clip-vit-large-patch14`
- **SigLIP**: Google的改进版CLIP模型
  - `google/siglip-base-patch16-224` (默认)
  - `google/siglip-large-patch16-256`

### NVIDIA NIM云端模型
- **nvidia/nvclip**: NVIDIA优化的CLIP模型
- **nvidia/nv-dinov2**: NVIDIA DINOv2视觉基础模型
- **nvidia/vila**: 多模态视觉-语言模型
- **meta/llama-3.2-90b-vision-instruct**: Llama视觉模型
- **meta/llama-3.2-11b-vision-instruct**: Llama视觉模型（小版本）

## 技术架构

- **图像编码**: CLIP/SigLIP/NVIDIA NIM模型提取图像特征
- **文本编码**: 同一模型的文本编码器处理查询文本
- **向量索引**: FAISS构建高效的相似度搜索索引
- **检索算法**: 余弦相似度匹配最相关的图片
- **云端推理**: 支持NVIDIA NIM API进行云端模型推理

## NVIDIA NIM使用说明

1. **获取API密钥**: 访问 [NVIDIA NGC](https://catalog.ngc.nvidia.com/) 获取API密钥
2. **设置环境变量**: `export NVIDIA_API_KEY="your_api_key"`
3. **选择模型**: 从支持的NVIDIA NIM模型中选择合适的模型
4. **构建索引**: 使用`--encoder_type nvidia_nim`参数

### NVIDIA NIM示例
```python
# 运行NVIDIA NIM示例
python examples/nvidia_nim_example.py

# 或者直接测试
export NVIDIA_API_KEY="your_api_key"
python test_system.py --encoder_type nvidia_nim --nvidia_api_key $NVIDIA_API_KEY
```

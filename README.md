# Text-to-Image Retrieval System

An image retrieval system based on CLIP/SigLIP + FAISS vector indexing that supports searching for related images using natural language descriptions.

## Features

- 🔍 **Text-to-Image Search**: Search for related images using natural language descriptions
- 🚀 **Efficient Retrieval**: Based on FAISS vector indexing for fast similarity search
- 🎯 **Multi-Model Support**: Supports CLIP, SigLIP, and NVIDIA NIM pretrained models
- 🌐 **Web Interface**: Provides intuitive Streamlit web interface
- 📊 **Visual Results**: Display search results and similarity scores
- ☁️ **Cloud Inference**: Supports NVIDIA NIM cloud vision-language model services

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Build Image Index

```bash
python build_index.py --image_dir ./images --index_path ./image_index.faiss
```

### 2. Start Web Interface

```bash
streamlit run app.py
```

### 3. Use API

```python
from src.retrieval import ImageRetrievalSystem

# Initialize system
retrieval_system = ImageRetrievalSystem()
retrieval_system.load_system("./image_index")

# Search images
results = retrieval_system.search("a cat sitting on a chair", top_k=5)
```

## Project Structure

```
image-retrieval/
├── src/
│   ├── encoders/          # Encoder modules
│   ├── indexing/          # Vector indexing modules
│   └── retrieval/         # Retrieval system core
├── examples/              # Example scripts
├── app.py                 # Streamlit web application
├── build_index.py         # Index building script
├── test_system.py         # System testing script
├── download_hf_dataset.py # Hugging Face dataset downloader
├── quick_start.py         # Quick start script
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Supported Models

### Local Models
- **CLIP**: OpenAI's classic vision-language model
  - `openai/clip-vit-base-patch32` (default)
  - `openai/clip-vit-large-patch14`
- **SigLIP**: Google's improved CLIP model
  - `google/siglip-base-patch16-224` (default)
  - `google/siglip-large-patch16-256`

### NVIDIA NIM Cloud Models
- **nvidia/nvclip**: NVIDIA optimized CLIP model
- **nvidia/nv-dinov2**: NVIDIA DINOv2 vision foundation model
- **nvidia/vila**: Multimodal vision-language model
- **meta/llama-3.2-90b-vision-instruct**: Llama vision model
- **meta/llama-3.2-11b-vision-instruct**: Llama vision model (smaller version)

## Technical Architecture

- **Image Encoding**: CLIP/SigLIP/NVIDIA NIM models extract image features
- **Text Encoding**: Same model's text encoder processes query text
- **Vector Indexing**: FAISS builds efficient similarity search index
- **Retrieval Algorithm**: Cosine similarity matching for most relevant images
- **Cloud Inference**: Supports NVIDIA NIM API for cloud model inference

## NVIDIA NIM Usage Guide

1. **Get API Key**: Visit [NVIDIA NGC](https://catalog.ngc.nvidia.com/) to obtain API key
2. **Set Environment Variable**: `export NVIDIA_API_KEY="your_api_key"`
3. **Choose Model**: Select appropriate model from supported NVIDIA NIM models
4. **Build Index**: Use `--encoder_type nvidia_nim` parameter

### NVIDIA NIM Examples
```bash
# Run NVIDIA NIM example
python examples/nvidia_nim_example.py

# Or test directly
export NVIDIA_API_KEY="your_api_key"
python test_system.py --encoder_type nvidia_nim --nvidia_api_key $NVIDIA_API_KEY
```

## Quick Start

### Method 1: One-click Start
```bash
python quick_start.py
```

### Method 2: Manual Steps

#### Download Sample Dataset
```bash
python download_hf_dataset.py --dataset simple --num_samples 50
```

#### Build Index with Different Encoders
```bash
# Using CLIP
python build_index.py --image_dir hf_images --index_path my_index --encoder_type clip

# Using SigLIP
python build_index.py --image_dir hf_images --index_path my_index --encoder_type siglip

# Using NVIDIA NIM
export NVIDIA_API_KEY="your_api_key"
python build_index.py --image_dir hf_images --index_path my_index --encoder_type nvidia_nim
```

#### Start Web Interface
```bash
streamlit run app.py -- --index_path my_index
```

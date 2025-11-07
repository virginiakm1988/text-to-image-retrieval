#!/usr/bin/env python3
"""
Test image retrieval system
"""
import os
import sys
import argparse
from PIL import Image
import requests
from io import BytesIO

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.retrieval import ImageRetrievalSystem


def download_sample_images():
    """Download some sample images for testing"""
    sample_urls = [
        ("https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400", "cat_1.jpg"),
        ("https://images.unsplash.com/photo-1573865526739-10659fec78a5?w=400", "cat_2.jpg"),
        ("https://images.unsplash.com/photo-1552053831-71594a27632d?w=400", "dog_1.jpg"),
        ("https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400", "dog_2.jpg"),
        ("https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d?w=400", "person_1.jpg"),
        ("https://images.unsplash.com/photo-1494790108755-2616b612b786?w=400", "person_2.jpg"),
        ("https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400", "mountain_1.jpg"),
        ("https://images.unsplash.com/photo-1464822759844-d150baec3e5e?w=400", "mountain_2.jpg"),
        ("https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400", "city_1.jpg"),
        ("https://images.unsplash.com/photo-1480714378408-67cf0d13bc1f?w=400", "city_2.jpg"),
    ]
    
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    print("Downloading sample images...")
    
    for url, filename in sample_urls:
        filepath = os.path.join(images_dir, filename)
        
        if os.path.exists(filepath):
            print(f"Skipping existing file: {filename}")
            continue
        
        try:
            print(f"Downloading: {filename}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # 保存图像
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
            img.save(filepath, 'JPEG', quality=85)
            
        except Exception as e:
            print(f"Download failed {filename}: {e}")
    
    print(f"Sample images saved to {images_dir} directory")
    return images_dir


def test_retrieval_system(encoder_type="clip", model_name=None, nvidia_api_key=None):
    """测试检索系统"""
    print("=" * 60)
    print("图像检索系统测试")
    print("=" * 60)
    
    # 下载示例图像
    images_dir = download_sample_images()
    
    # 初始化检索系统
    print(f"\n初始化检索系统 (编码器: {encoder_type})...")
    try:
        system = ImageRetrievalSystem(
            encoder_type=encoder_type,
            model_name=model_name,
            nvidia_api_key=nvidia_api_key
        )
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 构建索引
    print("\n构建图像索引...")
    try:
        added_count = system.add_images_from_directory(images_dir, batch_size=5)
        print(f"成功添加 {added_count} 张图像到索引")
    except Exception as e:
        print(f"构建索引失败: {e}")
        return
    
    # 显示系统统计
    stats = system.get_stats()
    print(f"\n系统统计:")
    print(f"- 总图像数量: {stats['total_images']}")
    print(f"- 嵌入向量维度: {stats['embedding_dim']}")
    print(f"- 编码器类型: {stats['encoder_type']}")
    
    # 测试文本搜索
    test_queries = [
        "a cat",
        "a dog",
        "a person",
        "mountains",
        "city skyline",
        "animal"
    ]
    
    print(f"\n测试文本搜索:")
    print("-" * 40)
    
    for query in test_queries:
        print(f"\n查询: '{query}'")
        try:
            results = system.search(query, top_k=3)
            
            if results:
                print("Results:")
                for i, result in enumerate(results, 1):
                    filename = os.path.basename(result['image_path'])
                    score = result['similarity_score']
                    print(f"  {i}. {filename} (similarity: {score:.3f})")
            else:
                print("  No results found")
                
        except Exception as e:
            print(f"  搜索失败: {e}")
    
    # Test image-to-image search
    print(f"\nTesting image-to-image search:")
    print("-" * 40)
    
    # Use first image as query
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if image_files:
        query_image_path = os.path.join(images_dir, image_files[0])
        print(f"\nUsing image query: {image_files[0]}")
        
        try:
            results = system.search_by_image(query_image_path, top_k=3)
            
            if results:
                print("Results:")
                for i, result in enumerate(results, 1):
                    filename = os.path.basename(result['image_path'])
                    score = result['similarity_score']
                    print(f"  {i}. {filename} (similarity: {score:.3f})")
            else:
                print("  No results found")
                
        except Exception as e:
            print(f"  Image search failed: {e}")
    
    # Save system
    index_path = "test_index"
    print(f"\nSaving system to: {index_path}")
    try:
        system.save_system(index_path)
        print("Save successful!")
        
        # Test loading
        print("\nTesting system loading...")
        new_system = ImageRetrievalSystem()
        new_system.load_system(index_path, nvidia_api_key)
        print("Load successful!")
        
        # Verify loaded system
        test_results = new_system.search("cat", top_k=2)
        print(f"Verification search: found {len(test_results)} results")
        
    except Exception as e:
        print(f"Save/load failed: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    
    # Provide next steps
    print(f"\nNext steps:")
    print(f"1. Start web interface:")
    print(f"   streamlit run app.py -- --index_path {index_path}")
    print(f"\n2. Or use Python API:")
    print(f"   from src.retrieval import ImageRetrievalSystem")
    print(f"   system = ImageRetrievalSystem()")
    print(f"   system.load_system('{index_path}')")
    print(f"   results = system.search('your query', top_k=5)")


def main():
    parser = argparse.ArgumentParser(description='Test image retrieval system')
    
    parser.add_argument('--encoder_type', type=str, default='clip',
                       choices=['clip', 'siglip', 'nvidia_nim'],
                       help='Encoder type')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name')
    parser.add_argument('--nvidia_api_key', type=str, default=None,
                       help='NVIDIA NIM API key')
    
    args = parser.parse_args()
    
    test_retrieval_system(
        encoder_type=args.encoder_type,
        model_name=args.model_name,
        nvidia_api_key=args.nvidia_api_key
    )


if __name__ == "__main__":
    main()

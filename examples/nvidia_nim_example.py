#!/usr/bin/env python3
"""
NVIDIA NIM编码器使用示例
"""
import os
import sys

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.encoders import NVIDIANIMEncoder
from src.retrieval import ImageRetrievalSystem


def test_nvidia_nim_encoder():
    """测试NVIDIA NIM编码器"""
    
    # 请设置您的NVIDIA NIM API密钥
    api_key = os.getenv('NVIDIA_API_KEY')  # 从环境变量获取
    
    if not api_key:
        print("请设置NVIDIA_API_KEY环境变量")
        print("或者直接在代码中设置api_key变量")
        return
    
    print("=" * 60)
    print("NVIDIA NIM编码器测试")
    print("=" * 60)
    
    # 可用的NVIDIA NIM模型
    available_models = [
        "nvidia/nvclip",           # NVIDIA CLIP模型
        "nvidia/nv-dinov2",        # NVIDIA DINOv2模型
        "nvidia/vila",             # VILA多模态模型
        "meta/llama-3.2-90b-vision-instruct",  # Llama视觉模型
        "meta/llama-3.2-11b-vision-instruct",  # Llama视觉模型（小版本）
    ]
    
    # 测试不同模型
    for model_name in available_models[:2]:  # 只测试前两个模型
        print(f"\n测试模型: {model_name}")
        print("-" * 40)
        
        try:
            # 初始化编码器
            encoder = NVIDIANIMEncoder(
                model_name=model_name,
                api_key=api_key
            )
            
            # 测试文本编码
            test_texts = [
                "a cat sitting on a chair",
                "a dog running in the park",
                "a beautiful mountain landscape"
            ]
            
            print("编码文本...")
            text_embeddings = encoder.encode_text(test_texts)
            print(f"文本嵌入形状: {text_embeddings.shape}")
            print(f"嵌入向量维度: {encoder.get_embedding_dim()}")
            
            # 如果有示例图像，测试图像编码
            images_dir = "../images"
            if os.path.exists(images_dir):
                image_files = [f for f in os.listdir(images_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if image_files:
                    print("编码图像...")
                    sample_images = [os.path.join(images_dir, f) for f in image_files[:3]]
                    image_embeddings = encoder.encode_images(sample_images)
                    print(f"图像嵌入形状: {image_embeddings.shape}")
                    
                    # 测试相似度计算
                    print("计算文本-图像相似度...")
                    for i, text in enumerate(test_texts):
                        similarities = encoder.compute_similarity(
                            text_embeddings[i], image_embeddings
                        )
                        print(f"'{text}' 与图像的相似度:")
                        for j, sim in enumerate(similarities):
                            print(f"  {image_files[j]}: {sim:.3f}")
            
            print(f"✅ {model_name} 测试成功")
            
        except Exception as e:
            print(f"❌ {model_name} 测试失败: {e}")


def test_nvidia_nim_retrieval_system():
    """测试使用NVIDIA NIM的完整检索系统"""
    
    api_key = os.getenv('NVIDIA_API_KEY')
    
    if not api_key:
        print("请设置NVIDIA_API_KEY环境变量")
        return
    
    print("\n" + "=" * 60)
    print("NVIDIA NIM检索系统测试")
    print("=" * 60)
    
    try:
        # 初始化检索系统
        system = ImageRetrievalSystem(
            encoder_type="nvidia_nim",
            model_name="nvidia/nvclip",
            nvidia_api_key=api_key
        )
        
        # 检查是否有示例图像
        images_dir = "../images"
        if not os.path.exists(images_dir):
            print("请先运行 test_system.py 下载示例图像")
            return
        
        # 构建索引
        print("构建图像索引...")
        added_count = system.add_images_from_directory(images_dir, batch_size=3)
        print(f"添加了 {added_count} 张图像")
        
        # 测试搜索
        test_queries = [
            "a cute cat",
            "a loyal dog", 
            "beautiful mountains",
            "urban cityscape"
        ]
        
        print("\n测试搜索:")
        for query in test_queries:
            print(f"\n查询: '{query}'")
            results = system.search(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                filename = os.path.basename(result['image_path'])
                score = result['similarity_score']
                print(f"  {i}. {filename} (相似度: {score:.3f})")
        
        # 保存系统
        index_path = "nvidia_nim_index"
        system.save_system(index_path)
        print(f"\n系统已保存到: {index_path}")
        
        print("\n✅ NVIDIA NIM检索系统测试成功!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")


def main():
    """主函数"""
    print("NVIDIA NIM 示例程序")
    print("请确保已设置 NVIDIA_API_KEY 环境变量")
    
    # 测试编码器
    test_nvidia_nim_encoder()
    
    # 测试完整检索系统
    test_nvidia_nim_retrieval_system()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)
    
    print("\n使用说明:")
    print("1. 设置API密钥: export NVIDIA_API_KEY='your_api_key'")
    print("2. 运行测试: python examples/nvidia_nim_example.py")
    print("3. 启动Web界面: streamlit run app.py -- --index_path nvidia_nim_index")


if __name__ == "__main__":
    main()


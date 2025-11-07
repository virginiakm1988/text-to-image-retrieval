#!/usr/bin/env python3
"""
Build image index script
"""
import argparse
import os
import sys
from src.retrieval import ImageRetrievalSystem


def main():
    parser = argparse.ArgumentParser(description='Build image retrieval index')
    
    # Required arguments
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Image directory path')
    parser.add_argument('--index_path', type=str, required=True,
                       help='Index save path (without extension)')
    
    # Optional arguments
    parser.add_argument('--encoder_type', type=str, default='clip',
                       choices=['clip', 'siglip', 'nvidia_nim'],
                       help='Encoder type')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name (use default if empty)')
    parser.add_argument('--index_type', type=str, default='flat',
                       choices=['flat', 'ivf', 'hnsw'],
                       help='FAISS index type')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                       help='Computing device (cuda/cpu)')
    parser.add_argument('--nvidia_api_key', type=str, default=None,
                       help='NVIDIA NIM API key')
    parser.add_argument('--recursive', action='store_true',
                       help='Recursively search subdirectories')
    
    args = parser.parse_args()
    
    # Check if image directory exists
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory does not exist: {args.image_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.dirname(args.index_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=" * 60)
    print("Image Retrieval Index Builder")
    print("=" * 60)
    print(f"Image directory: {args.image_dir}")
    print(f"Index path: {args.index_path}")
    print(f"Encoder type: {args.encoder_type}")
    print(f"Model name: {args.model_name or 'default'}")
    print(f"Index type: {args.index_type}")
    print(f"Batch size: {args.batch_size}")
    print(f"Recursive search: {'yes' if args.recursive else 'no'}")
    print("=" * 60)
    
    try:
        # Initialize retrieval system
        print("Initializing retrieval system...")
        retrieval_system = ImageRetrievalSystem(
            encoder_type=args.encoder_type,
            model_name=args.model_name,
            index_type=args.index_type,
            device=args.device,
            nvidia_api_key=args.nvidia_api_key
        )
        
        # Add images to index
        print("Starting to build index...")
        added_count = retrieval_system.add_images_from_directory(
            image_dir=args.image_dir,
            batch_size=args.batch_size,
            recursive=args.recursive
        )
        
        if added_count == 0:
            print("Warning: No image files found")
            sys.exit(1)
        
        # Save index
        print("Saving index...")
        retrieval_system.save_system(args.index_path)
        
        # Display statistics
        stats = retrieval_system.get_stats()
        print("\n" + "=" * 60)
        print("Index building completed!")
        print("=" * 60)
        print(f"Total images: {stats['total_images']}")
        print(f"Embedding dimension: {stats['embedding_dim']}")
        print(f"Index type: {stats['index_stats']['index_type']}")
        print(f"Index vectors: {stats['index_stats']['total_vectors']}")
        print("=" * 60)
        
        # Provide usage examples
        print("\nUsage examples:")
        print("1. Start web interface:")
        print(f"   streamlit run app.py -- --index_path {args.index_path}")
        print("\n2. Python API:")
        print("   from src.retrieval import ImageRetrievalSystem")
        print("   system = ImageRetrievalSystem()")
        print(f"   system.load_system('{args.index_path}')")
        print("   results = system.search('a cat sitting on a chair', top_k=5)")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

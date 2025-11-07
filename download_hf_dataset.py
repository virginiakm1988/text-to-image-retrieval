#!/usr/bin/env python3
"""
Download and prepare Hugging Face dataset for image retrieval
"""
import os
import sys
import argparse
from datasets import load_dataset
from PIL import Image
import json
from tqdm import tqdm


def download_cifar10_dataset(output_dir="hf_images", num_samples=100):
    """
    Download CIFAR-10 dataset from Hugging Face
    
    Args:
        output_dir: Directory to save images
        num_samples: Number of samples to download per class
    """
    print("Loading CIFAR-10 dataset from Hugging Face...")
    
    # Load CIFAR-10 dataset
    dataset = load_dataset("cifar10", split="train")
    
    # CIFAR-10 class names
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create subdirectories for each class
    for class_name in class_names:
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    
    # Download images
    class_counts = {i: 0 for i in range(10)}
    metadata = []
    
    print(f"Downloading {num_samples} samples per class...")
    
    for idx, sample in enumerate(tqdm(dataset)):
        label = sample['label']
        
        # Skip if we already have enough samples for this class
        if class_counts[label] >= num_samples:
            continue
        
        # Get image and class name
        image = sample['img']
        class_name = class_names[label]
        
        # Save image
        filename = f"{class_name}_{class_counts[label]:03d}.png"
        filepath = os.path.join(output_dir, class_name, filename)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image.save(filepath)
        
        # Store metadata
        metadata.append({
            'filename': filename,
            'filepath': filepath,
            'class': class_name,
            'label': label,
            'description': f"A {class_name} from CIFAR-10 dataset"
        })
        
        class_counts[label] += 1
        
        # Check if we have enough samples for all classes
        if all(count >= num_samples for count in class_counts.values()):
            break
    
    # Save metadata
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    total_images = sum(class_counts.values())
    print(f"\nDownloaded {total_images} images to {output_dir}/")
    print("Class distribution:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {class_counts[i]} images")
    
    return output_dir, total_images


def download_food101_dataset(output_dir="hf_images", num_samples=10):
    """
    Download Food-101 dataset from Hugging Face
    
    Args:
        output_dir: Directory to save images
        num_samples: Number of samples to download per class
    """
    print("Loading Food-101 dataset from Hugging Face...")
    
    try:
        # Load Food-101 dataset (smaller subset)
        dataset = load_dataset("food101", split="train")
        
        # Get unique food categories
        food_categories = set()
        for sample in dataset:
            food_categories.add(sample['label'])
        
        food_categories = sorted(list(food_categories))[:20]  # Take first 20 categories
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create subdirectories for each category
        for category in food_categories:
            category_dir = os.path.join(output_dir, str(category))
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)
        
        # Download images
        category_counts = {cat: 0 for cat in food_categories}
        metadata = []
        
        print(f"Downloading {num_samples} samples per category...")
        
        for idx, sample in enumerate(tqdm(dataset)):
            label = sample['label']
            
            # Skip if not in our selected categories or we have enough samples
            if label not in food_categories or category_counts[label] >= num_samples:
                continue
            
            # Get image
            image = sample['image']
            
            # Save image
            filename = f"food_{label}_{category_counts[label]:03d}.jpg"
            filepath = os.path.join(output_dir, str(label), filename)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(filepath, 'JPEG', quality=85)
            
            # Store metadata
            metadata.append({
                'filename': filename,
                'filepath': filepath,
                'category': str(label),
                'description': f"Food item from category {label}"
            })
            
            category_counts[label] += 1
            
            # Check if we have enough samples for all categories
            if all(count >= num_samples for count in category_counts.values()):
                break
        
        # Save metadata
        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        total_images = sum(category_counts.values())
        print(f"\nDownloaded {total_images} images to {output_dir}/")
        
        return output_dir, total_images
        
    except Exception as e:
        print(f"Error downloading Food-101: {e}")
        return None, 0


def download_imagenet_sample(output_dir="hf_images", num_samples=50):
    """
    Download a small sample from ImageNet-like dataset
    """
    print("Loading ImageNet sample dataset...")
    
    try:
        # Use a smaller ImageNet-like dataset
        dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        metadata = []
        count = 0
        
        print(f"Downloading {num_samples} sample images...")
        
        for sample in tqdm(dataset, total=num_samples):
            if count >= num_samples:
                break
            
            # Get image and label
            image = sample['image']
            label = sample['label']
            
            # Save image
            filename = f"imagenet_{count:04d}_label_{label}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(filepath, 'JPEG', quality=85)
            
            # Store metadata
            metadata.append({
                'filename': filename,
                'filepath': filepath,
                'label': label,
                'description': f"ImageNet sample with label {label}"
            })
            
            count += 1
        
        # Save metadata
        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDownloaded {count} images to {output_dir}/")
        return output_dir, count
        
    except Exception as e:
        print(f"Error downloading ImageNet sample: {e}")
        return None, 0


def download_simple_dataset(output_dir="hf_images", num_samples=50):
    """
    Download a simple, reliable dataset for testing
    """
    print("Loading a simple test dataset...")
    
    try:
        # Use CIFAR-10 as it's reliable and small
        dataset = load_dataset("cifar10", split="test")
        
        # CIFAR-10 class names
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        metadata = []
        
        print(f"Downloading {num_samples} sample images...")
        
        # Take a subset of the test set
        subset = dataset.select(range(min(num_samples, len(dataset))))
        
        for idx, sample in enumerate(tqdm(subset)):
            # Get image and label
            image = sample['img']
            label = sample['label']
            class_name = class_names[label]
            
            # Save image
            filename = f"{class_name}_{idx:03d}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(filepath)
            
            # Store metadata
            metadata.append({
                'filename': filename,
                'filepath': filepath,
                'class': class_name,
                'label': label,
                'description': f"A {class_name} from CIFAR-10 test set"
            })
        
        # Save metadata
        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDownloaded {len(metadata)} images to {output_dir}/")
        print("Sample classes included:")
        classes_in_sample = set(item['class'] for item in metadata)
        for class_name in sorted(classes_in_sample):
            count = sum(1 for item in metadata if item['class'] == class_name)
            print(f"  {class_name}: {count} images")
        
        return output_dir, len(metadata)
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None, 0


def main():
    parser = argparse.ArgumentParser(description='Download Hugging Face dataset for image retrieval')
    
    parser.add_argument('--dataset', type=str, default='simple',
                       choices=['cifar10', 'food101', 'imagenet', 'simple'],
                       help='Dataset to download')
    parser.add_argument('--output_dir', type=str, default='hf_images',
                       help='Output directory for images')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to download')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hugging Face Dataset Downloader")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print("=" * 60)
    
    # Download dataset
    if args.dataset == 'cifar10':
        output_dir, total_images = download_cifar10_dataset(args.output_dir, args.num_samples // 10)
    elif args.dataset == 'food101':
        output_dir, total_images = download_food101_dataset(args.output_dir, args.num_samples // 20)
    elif args.dataset == 'imagenet':
        output_dir, total_images = download_imagenet_sample(args.output_dir, args.num_samples)
    else:  # simple
        output_dir, total_images = download_simple_dataset(args.output_dir, args.num_samples)
    
    if output_dir and total_images > 0:
        print("\n" + "=" * 60)
        print("Download completed successfully!")
        print("=" * 60)
        print(f"Images saved to: {output_dir}")
        print(f"Total images: {total_images}")
        print(f"Metadata saved to: {os.path.join(output_dir, 'metadata.json')}")
        
        print("\nNext steps:")
        print("1. Build index:")
        print(f"   python build_index.py --image_dir {output_dir} --index_path hf_index")
        print("2. Start web interface:")
        print("   streamlit run app.py -- --index_path hf_index")
        print("3. Or run quick test:")
        print("   python test_system.py")
    else:
        print("Failed to download dataset")
        sys.exit(1)


if __name__ == "__main__":
    main()


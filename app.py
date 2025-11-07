#!/usr/bin/env python3
"""
Image Retrieval System Streamlit Web Interface
"""
import streamlit as st
import os
import sys
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import json

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.retrieval import ImageRetrievalSystem


def load_custom_css():
    """Load custom CSS styles"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    
    .result-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        background-color: #f9f9f9;
    }
    
    .similarity-score {
        background-color: #e8f4fd;
        color: #1f77b4;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .stats-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state"""
    if 'retrieval_system' not in st.session_state:
        st.session_state.retrieval_system = None
    if 'system_loaded' not in st.session_state:
        st.session_state.system_loaded = False
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []


def load_retrieval_system(index_path: str, nvidia_api_key: str = None):
    """Load retrieval system"""
    try:
        if st.session_state.retrieval_system is None:
            with st.spinner('Loading retrieval system...'):
                system = ImageRetrievalSystem()
                system.load_system(index_path, nvidia_api_key)
                st.session_state.retrieval_system = system
                st.session_state.system_loaded = True
                st.success('Retrieval system loaded successfully!')
        return st.session_state.retrieval_system
    except Exception as e:
        st.error(f'Failed to load retrieval system: {e}')
        return None


def display_search_results(results: List[Dict[str, Any]], query: str):
    """Display search results"""
    if not results:
        st.warning('No relevant images found')
        return
    
    st.markdown(f'<div class="sub-header">Search Results - "{query}"</div>', unsafe_allow_html=True)
    st.markdown(f'Found {len(results)} relevant images')
    
    # Create grid layout to display results
    cols_per_row = 3
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(results):
                result = results[i + j]
                
                with col:
                    # Display image
                    if os.path.exists(result['image_path']):
                        try:
                            img = Image.open(result['image_path'])
                            col.image(img, width=100)
                            
                            # Display similarity score
                            score = result['similarity_score']
                            col.markdown(
                                f'<div class="similarity-score">Similarity: {score:.3f}</div>',
                                unsafe_allow_html=True
                            )
                            
                            # Display file information
                            metadata = result.get('metadata', {})
                            filename = metadata.get('filename', os.path.basename(result['image_path']))
                            col.caption(f'File: {filename}')
                            
                            # Display image dimensions
                            if 'size' in metadata:
                                size = metadata['size']
                                col.caption(f'Size: {size[0]}×{size[1]}')
                            
                        except Exception as e:
                            col.error(f'Cannot load image: {e}')
                    else:
                        col.error('Image file does not exist')


def display_system_stats(system: ImageRetrievalSystem):
    """Display system statistics"""
    stats = system.get_stats()
    
    st.markdown('<div class="sub-header">System Information</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Images", stats['total_images'])
    
    with col2:
        st.metric("Embedding Dimension", stats['embedding_dim'])
    
    with col3:
        st.metric("Index Type", stats['index_stats']['index_type'].upper())
    
    # Detailed information
    with st.expander("Detailed Information"):
        st.json(stats)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_path', type=str, default=None,
                       help='Preloaded index path')
    parser.add_argument('--nvidia_api_key', type=str, default=None,
                       help='NVIDIA NIM API key')
    
    # Only parse arguments when running script directly
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        args = argparse.Namespace(index_path=None, nvidia_api_key=None)
    
    # Page configuration
    st.set_page_config(
        page_title="Text-to-Image Retrieval System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom styles
    load_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Main title
    st.markdown('<div class="main-header">🔍 Text-to-Image Retrieval System</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">Based on CLIP/SigLIP + FAISS Vector Index</div>', unsafe_allow_html=True)
    
    # Sidebar - System configuration
    with st.sidebar:
        st.header("System Configuration")
        
        # Index path input
        default_index = args.index_path or ""
        index_path = st.text_input(
            "Index Path",
            value=default_index,
            help="Enter the path to the built index file (without extension)"
        )
        
        # NVIDIA API key input
        nvidia_api_key = st.text_input(
            "NVIDIA NIM API Key",
            value=args.nvidia_api_key or "",
            type="password",
            help="Enter API key if using NVIDIA NIM encoder"
        )
        
        # 加载系统按钮
        if st.button("Load Retrieval System", type="primary"):
            if index_path:
                system = load_retrieval_system(index_path, nvidia_api_key)
                if system:
                    display_system_stats(system)
            else:
                st.error("Please enter index path")
        
        # 如果系统已加载，显示统计信息
        if st.session_state.system_loaded and st.session_state.retrieval_system:
            st.markdown("---")
            display_system_stats(st.session_state.retrieval_system)
    
    # Main interface
    if not st.session_state.system_loaded:
        # Instructions when system is not loaded
        st.info("👈 Please configure and load the retrieval system on the left")
        
        st.markdown("## Usage Instructions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 1. Build Index
            First, you need to build the image index:
            ```bash
            python build_index.py \\
                --image_dir ./images \\
                --index_path ./my_index \\
                --encoder_type clip
            ```
            """)
        
        with col2:
            st.markdown("""
            ### 2. Start Web Interface
            Then start the web interface:
            ```bash
            streamlit run app.py -- \\
                --index_path ./my_index
            ```
            """)
        
        st.markdown("""
        ### Supported Encoders
        - **CLIP**: OpenAI's classic vision-language model
        - **SigLIP**: Google's improved CLIP model
        - **NVIDIA NIM**: NVIDIA's cloud vision-language model service
        
        ### Supported Index Types
        - **Flat**: Exact search, suitable for small datasets
        - **IVF**: Inverted file index, suitable for large datasets
        - **HNSW**: Hierarchical navigable small world graph, suitable for high-dimensional data
        """)
        
    else:
        # System loaded, display search interface
        system = st.session_state.retrieval_system
        
        # Search interface
        st.markdown("## 🔍 Text Search")
        
        # Search input
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Enter search description",
                placeholder="e.g.: a cat sitting on a chair",
                help="Describe the image content you want to search for in natural language"
            )
        
        with col2:
            top_k = st.number_input("Number of results", min_value=1, max_value=50, value=9)
        
        # Search button
        if st.button("Search Images", type="primary") or query:
            if query.strip():
                with st.spinner('Searching...'):
                    try:
                        results = system.search(query.strip(), top_k=top_k)
                        st.session_state.search_results = results
                        display_search_results(results, query)
                    except Exception as e:
                        st.error(f'Search failed: {e}')
            else:
                st.warning("Please enter search description")
        
        # Display previous search results
        if st.session_state.search_results and not query:
            st.markdown("## Previous Search Results")
            display_search_results(st.session_state.search_results, "Previous search")
        
        # Image search functionality
        st.markdown("---")
        st.markdown("## 📷 Image-to-Image Search")
        
        uploaded_file = st.file_uploader(
            "Upload an image to search for similar images",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp']
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 2])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Query Image", width=250)
            
            with col2:
                if st.button("Search Similar Images", type="primary"):
                    with st.spinner('Searching for similar images...'):
                        try:
                            results = system.search_by_image(image, top_k=top_k)
                            display_search_results(results, "Similar images")
                        except Exception as e:
                            st.error(f'Image search failed: {e}')
        
        # Random image display
        st.markdown("---")
        st.markdown("## 🎲 Random Images")
        
        if st.button("Show Random Images"):
            try:
                random_paths = system.get_random_images(6)
                if random_paths:
                    cols = st.columns(3)
                    for i, path in enumerate(random_paths):
                        with cols[i % 3]:
                            if os.path.exists(path):
                                img = Image.open(path)
                                st.image(img, caption=os.path.basename(path), width=200)
            except Exception as e:
                st.error(f'Failed to get random images: {e}')


if __name__ == "__main__":
    main()

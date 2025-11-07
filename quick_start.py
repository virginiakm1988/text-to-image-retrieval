#!/usr/bin/env python3
"""
Quick start script - One-click experience for image retrieval system
"""
import os
import sys
import subprocess
import argparse


def install_dependencies():
    """Install dependency packages"""
    print("Installing dependency packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Dependency installation failed: {e}")
        return False


def run_test():
    """Run test script"""
    print("Running system test...")
    try:
        subprocess.check_call([sys.executable, "test_system.py"])
        print("✅ System test completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ System test failed: {e}")
        return False


def start_web_app():
    """Start web application"""
    print("Starting web interface...")
    try:
        # Check if test index exists
        if os.path.exists("test_index.faiss"):
            subprocess.check_call([
                sys.executable, "-m", "streamlit", "run", "app.py", 
                "--", "--index_path", "test_index"
            ])
        else:
            subprocess.check_call([sys.executable, "-m", "streamlit", "run", "app.py"])
    except subprocess.CalledProcessError as e:
        print(f"❌ Web interface startup failed: {e}")
    except KeyboardInterrupt:
        print("\n👋 Web interface closed")


def main():
    parser = argparse.ArgumentParser(description='Image retrieval system quick start')
    parser.add_argument('--skip-install', action='store_true',
                       help='Skip dependency installation')
    parser.add_argument('--skip-test', action='store_true',
                       help='Skip system test')
    parser.add_argument('--web-only', action='store_true',
                       help='Only start web interface')
    
    args = parser.parse_args()
    
    print("🔍 Image Retrieval System - Quick Start")
    print("=" * 50)
    
    if args.web_only:
        start_web_app()
        return
    
    # Install dependencies
    if not args.skip_install:
        if not install_dependencies():
            return
    
    # Run test
    if not args.skip_test:
        if not run_test():
            print("Test failed, but you can still manually start the web interface")
    
    # Ask whether to start web interface
    print("\n" + "=" * 50)
    response = input("Start web interface? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', '']:
        start_web_app()
    else:
        print("\nManually start web interface:")
        print("streamlit run app.py -- --index_path test_index")
        
        print("\nOr build custom index:")
        print("python build_index.py --image_dir ./your_images --index_path ./your_index")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Setup script for PeanutBot Training
Installs dependencies and prepares the environment
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    # Upgrade pip first
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def check_cuda():
    """Check CUDA availability"""
    print("🚀 Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available, will use CPU (training will be slower)")
        return True
    except ImportError:
        print("❌ PyTorch not installed yet")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ["models", "logs", "checkpoints"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory already exists: {directory}")
    
    return True

def download_model_info():
    """Show information about downloading TinyLlama"""
    print("\n📥 Model Download Information:")
    print("=" * 50)
    print("The training script will automatically download TinyLlama-1.1B-Chat-v1.0")
    print("This model is approximately 2.1 GB and will be downloaded on first run.")
    print("Download location: ~/.cache/huggingface/hub/")
    print("\nAlternative models you can use:")
    print("- TinyLlama/TinyLlama-1.1B-Chat-v1.0 (default, 2.1GB)")
    print("- TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T (2.1GB)")
    print("- TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T (2.1GB)")

def main():
    """Main setup function"""
    print("🥜 PeanutBot Training Setup 🥜")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        print("❌ Setup failed due to Python version incompatibility")
        return False
    
    # Create directories
    if not create_directories():
        print("❌ Setup failed while creating directories")
        return False
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed while installing requirements")
        return False
    
    # Check CUDA
    check_cuda()
    
    # Show model download info
    download_model_info()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Generate training data: python generate_data.py")
    print("2. Train the model: python train_with_data.py")
    print("3. Test the model: python inference.py")
    print("\n🥜 Happy training! 🥜")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

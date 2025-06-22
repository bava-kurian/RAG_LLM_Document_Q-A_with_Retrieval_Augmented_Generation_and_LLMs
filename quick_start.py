#!/usr/bin/env python3
"""
Quick start script for the RAG system.
This script helps users get up and running quickly with minimal setup.
"""

import os
import sys
from dotenv import load_dotenv

def check_environment():
    """Check if environment variables are set up."""
    print("🔍 Checking environment setup...")
    
    # Load environment variables
    load_dotenv()
    
    required_vars = [
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT", 
        "HUGGINGFACE_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n📝 Please create a .env file with the following variables:")
        print("PINECONE_API_KEY=your_pinecone_api_key")
        print("PINECONE_ENVIRONMENT=your_pinecone_environment")
        print("HUGGINGFACE_API_KEY=your_huggingface_api_key")
        return False
    
    print("✅ Environment variables are set up correctly")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        os.system("pip install -r requirements.txt")
        print("✅ Dependencies installed successfully")
        return True
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def run_basic_test():
    """Run a basic functionality test."""
    print("\n🧪 Running basic functionality test...")
    try:
        # Test imports
        from config import Config
        from document_loader import DocumentLoader
        from embeddings import EmbeddingGenerator
        
        print("✅ All modules imported successfully")
        
        # Test document loader
        loader = DocumentLoader()
        sample_docs = loader.create_sample_text()
        print(f"✅ Document loader created {len(sample_docs)} sample documents")
        
        # Test embeddings
        generator = EmbeddingGenerator()
        test_embedding = generator.generate_embedding("Test sentence")
        print(f"✅ Embeddings generated ({len(test_embedding)} dimensions)")
        
        return True
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def start_streamlit():
    """Start the Streamlit application."""
    print("\n🚀 Starting Streamlit application...")
    try:
        os.system("streamlit run app.py")
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

def main():
    """Main quick start function."""
    print("🚀 RAG System Quick Start")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        print("\n❌ Please set up your environment variables first.")
        print("See README.md for detailed instructions.")
        return
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies.")
        return
    
    # Run basic test
    if not run_basic_test():
        print("\n❌ Basic functionality test failed.")
        print("Please check your setup and try again.")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. The Streamlit app will start automatically")
    print("2. Load sample documents using the sidebar")
    print("3. Ask questions in the main interface")
    print("4. Press Ctrl+C to stop the application")
    
    # Ask user if they want to start the app
    response = input("\nWould you like to start the Streamlit app now? (y/n): ")
    if response.lower() in ['y', 'yes']:
        start_streamlit()
    else:
        print("\nTo start the app later, run: streamlit run app.py")

if __name__ == "__main__":
    main() 
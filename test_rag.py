#!/usr/bin/env python3
"""
Test script for the RAG system.
This script tests the basic functionality without requiring the full Streamlit interface.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    try:
        from config import Config
        Config.validate_config()
        print("✅ Configuration is valid")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_document_loader():
    """Test document loading functionality."""
    print("\nTesting document loader...")
    try:
        from document_loader import DocumentLoader
        
        loader = DocumentLoader()
        sample_docs = loader.create_sample_text()
        
        if sample_docs and len(sample_docs) > 0:
            print(f"✅ Document loader works - created {len(sample_docs)} sample documents")
            return True
        else:
            print("❌ Document loader failed - no documents created")
            return False
    except Exception as e:
        print(f"❌ Document loader error: {e}")
        return False

def test_embeddings():
    """Test embedding generation."""
    print("\nTesting embeddings...")
    try:
        from embeddings import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        test_text = "This is a test sentence."
        embedding = generator.generate_embedding(test_text)
        
        if embedding and len(embedding) > 0:
            print(f"✅ Embeddings work - generated {len(embedding)}-dimensional embedding")
            return True
        else:
            print("❌ Embeddings failed - no embedding generated")
            return False
    except Exception as e:
        print(f"❌ Embeddings error: {e}")
        return False

def test_vector_store():
    """Test vector store operations."""
    print("\nTesting vector store...")
    try:
        from vector_store import VectorStore
        
        vector_store = VectorStore()
        stats = vector_store.get_index_stats()
        
        if stats:
            print("✅ Vector store works - connected to Pinecone")
            return True
        else:
            print("❌ Vector store failed - no stats retrieved")
            return False
    except Exception as e:
        print(f"❌ Vector store error: {e}")
        return False

def test_llm_handler():
    """Test LLM handler."""
    print("\nTesting LLM handler...")
    try:
        from llm_handler import LLMHandler
        
        # Use a smaller model for testing
        handler = LLMHandler("microsoft/DialoGPT-small")
        test_prompt = "Hello, how are you?"
        response = handler.generate_response(test_prompt)
        
        if response and len(response) > 0:
            print("✅ LLM handler works - generated response")
            return True
        else:
            print("❌ LLM handler failed - no response generated")
            return False
    except Exception as e:
        print(f"❌ LLM handler error: {e}")
        return False

def test_full_rag_system():
    """Test the complete RAG system."""
    print("\nTesting full RAG system...")
    try:
        from rag_system import RAGSystem
        
        # Initialize RAG system
        rag = RAGSystem()
        
        # Load sample documents
        rag.load_and_index_documents(use_sample=True)
        
        # Test question answering
        question = "What is artificial intelligence?"
        result = rag.answer_question(question, k=3)
        
        if result and result.get("answer"):
            print("✅ Full RAG system works - answered question successfully")
            print(f"Question: {result['question']}")
            print(f"Answer: {result['answer'][:200]}...")
            return True
        else:
            print("❌ Full RAG system failed - no answer generated")
            return False
    except Exception as e:
        print(f"❌ Full RAG system error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running RAG System Tests")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_config),
        ("Document Loader", test_document_loader),
        ("Embeddings", test_embeddings),
        ("Vector Store", test_vector_store),
        ("LLM Handler", test_llm_handler),
        ("Full RAG System", test_full_rag_system)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The RAG system is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
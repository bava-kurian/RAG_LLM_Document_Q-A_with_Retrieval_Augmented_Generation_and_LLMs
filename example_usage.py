#!/usr/bin/env python3
"""
Example usage of the RAG system.
This script demonstrates how to use the RAG system programmatically.
"""

import os
from dotenv import load_dotenv
from rag_system import RAGSystem

def main():
    """Example usage of the RAG system."""
    
    # Load environment variables
    load_dotenv()
    
    print("🤖 RAG System Example Usage")
    print("=" * 50)
    
    try:
        # Initialize the RAG system
        print("Initializing RAG system...")
        rag = RAGSystem()
        
        # Load sample documents
        print("\nLoading sample documents...")
        rag.load_and_index_documents(use_sample=True)
        
        # Example questions to test
        questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are vector databases used for?",
            "What is the relationship between AI and NLP?",
            "How do neural networks work in deep learning?"
        ]
        
        print("\n" + "=" * 50)
        print("🔍 Testing Question Answering")
        print("=" * 50)
        
        # Ask questions and get answers
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}: {question}")
            print("-" * 40)
            
            # Get answer from RAG system
            result = rag.answer_question(question, k=3)
            
            # Display answer
            print(f"Answer: {result['answer']}")
            print(f"Documents used: {result.get('num_docs_used', 0)}")
            
            # Show a snippet of the first context document
            if result.get('context'):
                print(f"Context snippet: {result['context'][0][:200]}...")
            
            print()
        
        # Get system information
        print("=" * 50)
        print("ℹ️ System Information")
        print("=" * 50)
        
        system_info = rag.get_system_info()
        print(f"Embedding Model: {system_info['embedding_model']}")
        print(f"LLM Model: {system_info['llm_model']}")
        print(f"Chunk Size: {system_info['chunk_size']}")
        print(f"Chunk Overlap: {system_info['chunk_overlap']}")
        
        # Vector store stats
        if 'vector_store_stats' in system_info:
            stats = system_info['vector_store_stats']
            if 'total_vector_count' in stats:
                print(f"Total vectors in index: {stats['total_vector_count']}")
        
        print("\n✅ Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Set up your environment variables (.env file)")
        print("2. Installed all dependencies (pip install -r requirements.txt)")
        print("3. Valid API keys for Pinecone and Hugging Face")

def interactive_mode():
    """Interactive mode for asking custom questions."""
    
    # Load environment variables
    load_dotenv()
    
    print("🤖 RAG System Interactive Mode")
    print("=" * 50)
    
    try:
        # Initialize the RAG system
        print("Initializing RAG system...")
        rag = RAGSystem()
        
        # Load sample documents
        print("Loading sample documents...")
        rag.load_and_index_documents(use_sample=True)
        
        print("\n✅ System ready! Type 'quit' to exit.")
        print("-" * 50)
        
        while True:
            # Get user question
            question = input("\n❓ Enter your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                print("Please enter a question.")
                continue
            
            try:
                # Get answer
                print("🔍 Searching for answer...")
                result = rag.answer_question(question, k=3)
                
                # Display answer
                print(f"\n💡 Answer: {result['answer']}")
                print(f"📚 Documents used: {result.get('num_docs_used', 0)}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main() 
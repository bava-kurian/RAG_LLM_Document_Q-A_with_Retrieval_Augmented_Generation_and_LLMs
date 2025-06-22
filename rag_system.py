from typing import List, Optional
from document_loader import DocumentLoader
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from llm_handler import LLMHandler
from config import Config

class RAGSystem:
    """Main RAG system that orchestrates all components."""
    
    def __init__(self):
        """Initialize the RAG system with all components."""
        # Validate configuration
        Config.validate_config()
        
        # Initialize components
        self.document_loader = DocumentLoader(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.embedding_generator = EmbeddingGenerator(Config.EMBEDDING_MODEL)
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()
        
        print("RAG System initialized successfully")
    
    def load_and_index_documents(self, file_path: str = None, directory_path: str = None, use_sample: bool = False) -> None:
        """Load documents and index them in the vector store."""
        try:
            documents = []
            
            if use_sample:
                print("Loading sample documents...")
                documents = self.document_loader.create_sample_text()
            elif file_path:
                print(f"Loading document: {file_path}")
                documents = self.document_loader.load_document(file_path)
            elif directory_path:
                print(f"Loading documents from directory: {directory_path}")
                documents = self.document_loader.load_documents_from_directory(directory_path)
            else:
                raise ValueError("Must specify file_path, directory_path, or use_sample=True")
            
            if not documents:
                print("No documents loaded")
                return
            
            print(f"Loaded {len(documents)} document chunks")
            
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            print("Documents indexed successfully")
            
        except Exception as e:
            print(f"Error loading and indexing documents: {str(e)}")
            raise
    
    def answer_question(self, question: str, k: int = 5) -> dict:
        """Answer a question using the RAG system."""
        try:
            # Search for relevant documents
            print(f"Searching for relevant documents for: {question}")
            relevant_docs = self.vector_store.similarity_search(question, k=k)
            
            if not relevant_docs:
                return {
                    "answer": "No relevant documents found to answer your question.",
                    "context": [],
                    "question": question
                }
            
            print(f"Found {len(relevant_docs)} relevant documents")
            
            # Generate answer using LLM
            print("Generating answer...")
            answer = self.llm_handler.answer_question_with_context(question, relevant_docs)
            
            # Prepare context for display
            context = [doc.page_content for doc in relevant_docs]
            
            return {
                "answer": answer,
                "context": context,
                "question": question,
                "num_docs_used": len(relevant_docs)
            }
            
        except Exception as e:
            print(f"Error answering question: {str(e)}")
            return {
                "answer": f"Error answering question: {str(e)}",
                "context": [],
                "question": question
            }
    
    def get_system_info(self) -> dict:
        """Get information about the RAG system."""
        try:
            # Get vector store stats
            vector_stats = self.vector_store.get_index_stats()
            
            # Get model info
            model_info = self.llm_handler.get_model_info()
            
            return {
                "embedding_model": self.embedding_generator.model_name,
                "llm_model": model_info["model_name"],
                "vector_store_stats": vector_stats,
                "chunk_size": Config.CHUNK_SIZE,
                "chunk_overlap": Config.CHUNK_OVERLAP
            }
            
        except Exception as e:
            print(f"Error getting system info: {str(e)}")
            return {"error": str(e)}
    
    def clear_index(self) -> None:
        """Clear the vector store index."""
        try:
            self.vector_store.delete_index()
            print("Vector store index cleared")
        except Exception as e:
            print(f"Error clearing index: {str(e)}")
            raise 
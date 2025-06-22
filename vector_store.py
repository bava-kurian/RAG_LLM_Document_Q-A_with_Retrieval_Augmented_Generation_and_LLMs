import pinecone
from typing import List, Dict, Any
from langchain.schema import Document
from langchain_community.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from config import Config

class VectorStore:
    """Handles Pinecone vector store operations."""
    
    def __init__(self, index_name: str = None):
        self.index_name = index_name or Config.PINECONE_INDEX_NAME
        self.pinecone_api_key = Config.PINECONE_API_KEY
        self.pinecone_environment = Config.PINECONE_ENVIRONMENT
        
        # Initialize Pinecone
        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_environment)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize or get the index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the Pinecone index."""
        try:
            # Check if index exists
            if self.index_name not in pinecone.list_indexes():
                # Get embedding dimension
                dummy_embedding = self.embeddings.embed_query("test")
                dimension = len(dummy_embedding)
                
                # Create index
                pinecone.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine"
                )
                print(f"Created new Pinecone index: {self.index_name}")
            else:
                print(f"Using existing Pinecone index: {self.index_name}")
            
            # Get the index
            self.index = pinecone.Index(self.index_name)
            
        except Exception as e:
            print(f"Error initializing Pinecone index: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        try:
            # Create vector store
            vectorstore = Pinecone.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=self.index_name
            )
            print(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search on the vector store."""
        try:
            # Create vector store instance for searching
            vectorstore = Pinecone.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings
            )
            
            # Perform similarity search
            results = vectorstore.similarity_search(query, k=k)
            return results
            
        except Exception as e:
            print(f"Error performing similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Perform similarity search and return documents with scores."""
        try:
            # Create vector store instance for searching
            vectorstore = Pinecone.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings
            )
            
            # Perform similarity search with scores
            results = vectorstore.similarity_search_with_score(query, k=k)
            return results
            
        except Exception as e:
            print(f"Error performing similarity search with scores: {str(e)}")
            raise
    
    def delete_index(self) -> None:
        """Delete the Pinecone index."""
        try:
            if self.index_name in pinecone.list_indexes():
                pinecone.delete_index(self.index_name)
                print(f"Deleted Pinecone index: {self.index_name}")
            else:
                print(f"Index {self.index_name} does not exist")
                
        except Exception as e:
            print(f"Error deleting index: {str(e)}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            print(f"Error getting index stats: {str(e)}")
            raise 
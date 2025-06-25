import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import Document
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import Config


class VectorStore:
    """Handles Pinecone vector store operations with the latest Pinecone client (v3)."""

    def __init__(self, index_name: str = None, embedding_model: str = None):
        """
        Initialize the vector store.
        
        Args:
            index_name: Name of the Pinecone index (defaults to Config.PINECONE_INDEX_NAME)
            embedding_model: Name of the HuggingFace embedding model
        """
        self.index_name = index_name or Config.PINECONE_INDEX_NAME
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL

        # Initialize Pinecone client (v3 style)
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize or connect to index
        self.index = self._initialize_index()

    def _initialize_index(self):
        """Initialize or connect to the Pinecone index."""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                # Get embedding dimension from test query
                dummy_embedding = self.embeddings.embed_query("test")
                dimension = len(dummy_embedding)

                # Create new index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-west-1"  # Adjust as needed
                    )
                )
                print(f"✅ Created new Pinecone index: {self.index_name}")
            else:
                print(f"✅ Using existing Pinecone index: {self.index_name}")

            # Connect to the index
            return self.pc.Index(self.index_name)

        except Exception as e:
            print(f"❌ Error initializing Pinecone index: {str(e)}")
            raise

    def add_documents(self, documents: List[Document], namespace: Optional[str] = None) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects
            namespace: Optional namespace for the vectors
        """
        try:
            LangchainPinecone.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=self.index_name,
                namespace=namespace
            )
            print(f"✅ Added {len(documents)} documents to index '{self.index_name}'")
        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            raise

    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        namespace: Optional[str] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: The query string
            k: Number of results to return
            namespace: Optional namespace to search in
            kwargs: Additional arguments for the search
            
        Returns:
            List of matching documents
        """
        try:
            vectorstore = LangchainPinecone.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace=namespace
            )
            return vectorstore.similarity_search(query, k=k, **kwargs)
        except Exception as e:
            print(f"❌ Error in similarity search: {str(e)}")
            raise

    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5, 
        namespace: Optional[str] = None,
        **kwargs
    ) -> List[tuple[Document, float]]:
        """
        Perform similarity search and return documents with scores.
        
        Args:
            query: The query string
            k: Number of results to return
            namespace: Optional namespace to search in
            kwargs: Additional arguments for the search
            
        Returns:
            List of tuples containing (document, score)
        """
        try:
            vectorstore = LangchainPinecone.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace=namespace
            )
            return vectorstore.similarity_search_with_score(query, k=k, **kwargs)
        except Exception as e:
            print(f"❌ Error in similarity search with scores: {str(e)}")
            raise

    def delete_index(self) -> None:
        """Delete the Pinecone index."""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            if self.index_name in existing_indexes:
                self.pc.delete_index(self.index_name)
                print(f"🗑️ Deleted Pinecone index: {self.index_name}")
            else:
                print(f"ℹ️ Index {self.index_name} does not exist")
        except Exception as e:
            print(f"❌ Error deleting index: {str(e)}")
            raise

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary containing index statistics
        """
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            print(f"❌ Error getting index stats: {str(e)}")
            raise

    def clear_namespace(self, namespace: str) -> None:
        """
        Clear all vectors in a namespace.
        
        Args:
            namespace: The namespace to clear
        """
        try:
            self.index.delete(delete_all=True, namespace=namespace)
            print(f"🧹 Cleared all vectors in namespace: {namespace}")
        except Exception as e:
            print(f"❌ Error clearing namespace: {str(e)}")
            raise
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

class EmbeddingGenerator:
    """Handles generation of embeddings using Hugging Face models."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Use CPU for free tier
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        # Create a dummy embedding to get the dimension
        dummy_embedding = self.generate_embedding("test")
        return len(dummy_embedding)
    
    def prepare_documents_for_embedding(self, documents: List[Document]) -> List[str]:
        """Extract text content from documents for embedding generation."""
        texts = []
        for doc in documents:
            texts.append(doc.page_content)
        return texts 
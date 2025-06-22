import os
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

class DocumentLoader:
    """Handles loading and splitting of documents."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document and split it into chunks."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Determine loader based on file extension
        file_extension = file_path.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension in ['txt', 'md']:
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Load and split the document
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        return chunks
    
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory."""
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        all_chunks = []
        supported_extensions = {'.pdf', '.txt', '.md'}
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                file_extension = os.path.splitext(filename)[1].lower()
                if file_extension in supported_extensions:
                    try:
                        chunks = self.load_document(file_path)
                        all_chunks.extend(chunks)
                        print(f"Loaded {len(chunks)} chunks from {filename}")
                    except Exception as e:
                        print(f"Error loading {filename}: {str(e)}")
        
        return all_chunks
    
    def create_sample_text(self) -> List[Document]:
        """Create sample text documents for testing."""
        sample_texts = [
            """
            Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
            that work and react like humans. Some of the activities computers with artificial intelligence are 
            designed for include speech recognition, learning, planning, and problem solving.
            
            Machine learning is a subset of AI that enables computers to learn and improve from experience 
            without being explicitly programmed. It focuses on developing computer programs that can access 
            data and use it to learn for themselves.
            
            Deep learning is a subset of machine learning that uses neural networks with multiple layers 
            to model and understand complex patterns in data. It has been particularly successful in areas 
            like image recognition, natural language processing, and speech recognition.
            """,
            
            """
            Natural Language Processing (NLP) is a field of AI that focuses on the interaction between 
            computers and human language. It involves developing algorithms and models that can understand, 
            interpret, and generate human language in a way that is both meaningful and useful.
            
            Key applications of NLP include machine translation, sentiment analysis, chatbots, 
            text summarization, and question answering systems. These applications are becoming 
            increasingly important in our digital world.
            
            Recent advances in NLP have been driven by large language models like GPT, BERT, and 
            their successors, which have achieved remarkable performance on various language tasks.
            """,
            
            """
            Vector databases are specialized databases designed to store and retrieve high-dimensional 
            vector data efficiently. They are particularly useful for applications involving similarity 
            search, recommendation systems, and AI/ML workloads.
            
            Pinecone is a popular vector database service that provides a managed solution for 
            storing and querying vector embeddings. It offers features like real-time similarity 
            search, automatic scaling, and integration with popular ML frameworks.
            
            Vector databases are essential for modern AI applications, especially those involving 
            semantic search, recommendation systems, and retrieval-augmented generation (RAG) systems.
            """
        ]
        
        documents = []
        for i, text in enumerate(sample_texts):
            doc = Document(
                page_content=text.strip(),
                metadata={"source": f"sample_document_{i+1}", "type": "sample"}
            )
            documents.append(doc)
        
        # Split the documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        return chunks 
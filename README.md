# RAG Document Q&A System

A complete Retrieval-Augmented Generation (RAG) application built with modern AI technologies. This system allows you to load documents, create embeddings, store them in a vector database, and answer questions using a large language model.

## 🚀 Features

- **Document Processing**: Load and split PDF, TXT, and MD files
- **Embeddings**: Generate embeddings using Hugging Face models
- **Vector Storage**: Store and query documents using Pinecone
- **LLM Integration**: Answer questions using Hugging Face language models
- **Modern UI**: Beautiful Streamlit interface with real-time interactions
- **Modular Design**: Clean, maintainable code structure

## 🛠️ Technology Stack

- **Streamlit**: Web interface
- **LangChain**: Framework for LLM applications
- **Hugging Face**: Embeddings and language models
- **Pinecone**: Vector database
- **Transformers**: Model loading and inference
- **Sentence Transformers**: Text embeddings

## 📋 Prerequisites

- Python 3.8+
- Pinecone account (free tier available)
- Hugging Face account (free)

## 🔧 Installation

1. **Clone the repository**

   ```bash
   git clone [<repository-url>](https://github.com/bava-kurian/RAG_LLM_Document_Q-A_with_Retrieval_Augmented_Generation_and_LLMs)
   cd RAG_LLM_Document_Q-A_with_Retrieval_Augmented_Generation_and_LLMs
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file in the project root:

   ```bash
   # Pinecone API Configuration
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=your_pinecone_environment_here
   PINECONE_INDEX_NAME=rag-documents

   # Hugging Face Configuration
   HUGGINGFACE_API_KEY=your_huggingface_api_key_here

   # Model Configuration
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   LLM_MODEL=microsoft/DialoGPT-small
   ```

## 🔑 API Keys Setup

### Pinecone Setup

1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Create a free account
3. Create a new index (use "cosine" metric)
4. Copy your API key and environment

### Hugging Face Setup

1. Go to [Hugging Face](https://huggingface.co/)
2. Create an account
3. Go to Settings → Access Tokens
4. Create a new token with read permissions

## 🚀 Usage

1. **Start the application**

   ```bash
   streamlit run app.py
   ```

2. **Load documents**

   - Use sample documents (recommended for testing)
   - Upload a single file (PDF, TXT, MD)
   - Load from a directory

3. **Ask questions**
   - Enter your question in the text area
   - Adjust the number of documents to retrieve
   - Click "Get Answer" to see the response

## 📁 Project Structure

```
RAG-LLM/
├── app.py                 # Main Streamlit application
├── rag_system.py          # Main RAG system orchestrator
├── config.py              # Configuration management
├── document_loader.py     # Document loading and splitting
├── embeddings.py          # Embedding generation
├── vector_store.py        # Pinecone vector store operations
├── llm_handler.py         # Language model handling
├── requirements.txt       # Python dependencies
├── env_example.txt        # Environment variables template
├── test_rag.py           # Testing script
├── quick_start.py        # Quick setup script
├── example_usage.py      # Example usage script
└── README.md             # This file
```

## 🔧 Configuration

### Model Configuration

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (default)
- **LLM Model**: `microsoft/DialoGPT-small` (default, optimized for free tier)
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters

### Customization

You can modify the models and parameters in the `config.py` file or through environment variables.

**Note**: For production use, you can change the LLM model to larger models like:

- `mistralai/Mistral-7B-Instruct-v0.2`
- `tiiuae/falcon-7b-instruct`
- `microsoft/DialoGPT-medium` or `microsoft/DialoGPT-large`

## 💡 Example Questions

After loading the sample documents, try these questions:

- "What is artificial intelligence?"
- "How does machine learning work?"
- "What are vector databases used for?"
- "What is the relationship between AI and NLP?"

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**

   - Ensure you have sufficient RAM (at least 4GB recommended for default model)
   - The default model (`microsoft/DialoGPT-small`) is optimized for free tier usage
   - Try using a smaller model if memory is limited

2. **Pinecone Connection Issues**

   - Verify your API key and environment
   - Check your internet connection
   - Ensure your Pinecone index exists
   - The application uses Pinecone API v3 - make sure you're using the latest client

3. **Memory Issues**

   - Reduce chunk size in `config.py`
   - Use a smaller embedding model
   - Close other applications to free up memory
   - The default model is optimized for low-memory environments

4. **Accelerate Dependency Issues**
   - The application includes `accelerate>=0.26.0` in requirements
   - If you encounter accelerate-related errors, try: `pip install accelerate>=0.26.0`

### Performance Tips

- Use GPU if available (modify `device_map` in `llm_handler.py`)
- Adjust chunk size based on your document characteristics
- Use appropriate number of documents for retrieval (k parameter)
- The default model is optimized for quick responses on CPU

### Testing

Run the test script to verify your setup:

```bash
python test_rag.py
```

For quick setup and testing:

```bash
python quick_start.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for providing excellent open-source models
- Pinecone for the vector database service
- LangChain for the framework
- Streamlit for the web interface

## 📞 Support

If you encounter any issues or have questions, please:

1. Check the troubleshooting section
2. Run the test script: `python test_rag.py`
3. Search existing issues
4. Create a new issue with detailed information

---

**Note**: This application uses free-tier services and is optimized for low-resource environments. Be mindful of rate limits and usage quotas.

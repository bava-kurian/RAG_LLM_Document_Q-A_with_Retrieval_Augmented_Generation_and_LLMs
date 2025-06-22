import streamlit as st
import os
from rag_system import RAGSystem
import time

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #b3d9ff;
        margin: 1rem 0;
    }
    .context-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #cce5ff;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_rag_system():
    """Initialize the RAG system."""
    try:
        if 'rag_system' not in st.session_state:
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_system = RAGSystem()
        return st.session_state.rag_system
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        st.error("Please check your environment variables and API keys.")
        return None

def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">🤖 RAG Document Q&A System</h1>', unsafe_allow_html=True)
    
    # Initialize RAG system
    rag_system = initialize_rag_system()
    if not rag_system:
        st.stop()
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown('<h3 class="sub-header">⚙️ Configuration</h3>', unsafe_allow_html=True)
        
        # Document loading section
        st.subheader("📚 Load Documents")
        load_option = st.selectbox(
            "Choose document loading option:",
            ["Use Sample Documents", "Upload Single File", "Load from Directory"]
        )
        
        if load_option == "Use Sample Documents":
            if st.button("Load Sample Documents"):
                with st.spinner("Loading sample documents..."):
                    try:
                        rag_system.load_and_index_documents(use_sample=True)
                        st.success("Sample documents loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading sample documents: {str(e)}")
        
        elif load_option == "Upload Single File":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'pdf', 'md']
            )
            if uploaded_file is not None and st.button("Load File"):
                with st.spinner("Loading document..."):
                    try:
                        # Save uploaded file temporarily
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        rag_system.load_and_index_documents(file_path=temp_path)
                        st.success("Document loaded successfully!")
                        
                        # Clean up temp file
                        os.remove(temp_path)
                    except Exception as e:
                        st.error(f"Error loading document: {str(e)}")
        
        elif load_option == "Load from Directory":
            directory_path = st.text_input("Enter directory path:")
            if directory_path and st.button("Load from Directory"):
                with st.spinner("Loading documents from directory..."):
                    try:
                        rag_system.load_and_index_documents(directory_path=directory_path)
                        st.success("Documents loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading documents: {str(e)}")
        
        # System info
        st.subheader("ℹ️ System Info")
        if st.button("Show System Info"):
            try:
                info = rag_system.get_system_info()
                st.json(info)
            except Exception as e:
                st.error(f"Error getting system info: {str(e)}")
        
        # Clear index
        st.subheader("🗑️ Management")
        if st.button("Clear Index"):
            try:
                rag_system.clear_index()
                st.success("Index cleared successfully!")
            except Exception as e:
                st.error(f"Error clearing index: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h3 class="sub-header">❓ Ask Questions</h3>', unsafe_allow_html=True)
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="Ask anything about your documents..."
        )
        
        # Number of documents to retrieve
        k_docs = st.slider("Number of documents to retrieve:", min_value=1, max_value=10, value=5)
        
        # Submit button
        if st.button("🔍 Get Answer", type="primary"):
            if question.strip():
                with st.spinner("Searching for answer..."):
                    try:
                        # Get answer from RAG system
                        result = rag_system.answer_question(question, k=k_docs)
                        
                        # Display answer
                        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                        st.markdown("### Answer:")
                        st.write(result["answer"])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Store result in session state for context display
                        st.session_state.last_result = result
                        
                    except Exception as e:
                        st.error(f"Error getting answer: {str(e)}")
            else:
                st.warning("Please enter a question.")
    
    with col2:
        st.markdown('<h3 class="sub-header">📖 Context</h3>', unsafe_allow_html=True)
        
        # Display context from last query
        if 'last_result' in st.session_state and st.session_state.last_result.get("context"):
            result = st.session_state.last_result
            
            st.markdown(f"**Question:** {result['question']}")
            st.markdown(f"**Documents used:** {result.get('num_docs_used', 0)}")
            
            st.markdown("**Relevant Context:**")
            for i, context in enumerate(result["context"], 1):
                with st.expander(f"Document {i}"):
                    st.markdown('<div class="context-box">', unsafe_allow_html=True)
                    st.write(context[:500] + "..." if len(context) > 500 else context)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Ask a question to see relevant context here.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            Built with Streamlit, LangChain, Hugging Face, and Pinecone
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 
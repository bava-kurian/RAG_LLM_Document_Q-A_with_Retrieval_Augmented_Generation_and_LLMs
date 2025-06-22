from typing import List, Optional
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import Config

class LLMHandler:
    """Handles Hugging Face language models for text generation."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.LLM_MODEL
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.llm = None
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Hugging Face model and pipeline."""
        try:
            print(f"Loading model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto"
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Create LangChain LLM
            self.llm = HuggingFacePipeline(pipeline=self.pipeline)
            
            print(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {str(e)}")
            raise
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response using the language model."""
        try:
            response = self.llm(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def create_rag_prompt(self, question: str, context_docs: List[Document]) -> str:
        """Create a prompt for RAG-based question answering."""
        # Extract context from documents
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Create the prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a helpful AI assistant. Use the following context to answer the question.
            If the context doesn't contain enough information to answer the question, say so.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
        )
        
        # Format the prompt
        prompt = prompt_template.format(context=context, question=question)
        return prompt
    
    def answer_question_with_context(self, question: str, context_docs: List[Document]) -> str:
        """Answer a question using provided context documents."""
        try:
            # Create RAG prompt
            prompt = self.create_rag_prompt(question, context_docs)
            
            # Generate response
            response = self.generate_response(prompt)
            return response
            
        except Exception as e:
            print(f"Error answering question: {str(e)}")
            return f"Error answering question: {str(e)}"
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "tokenizer": type(self.tokenizer).__name__ if self.tokenizer else None,
            "model": type(self.model).__name__ if self.model else None,
            "pipeline": type(self.pipeline).__name__ if self.pipeline else None
        } 
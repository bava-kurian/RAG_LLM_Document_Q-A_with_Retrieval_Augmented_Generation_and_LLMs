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
        self.max_length = 1024  # Set max_length for prompt truncation
        
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
            
            # Load model without device_map for free tier compatibility
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
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
    
    def safe_prompt(self, prompt: str) -> str:
        tokens = self.tokenizer.encode(prompt, truncation=True, max_length=self.max_length)
        # If truncation occurred, warn or log
        if len(tokens) == self.max_length:
            print("⚠️ Prompt was truncated to fit the model's max token length.")
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response using the language model."""
        try:
            safe_prompt = self.safe_prompt(prompt)
            response = self.llm.invoke(safe_prompt)
            return response.strip()
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def safe_rag_prompt(self, question: str, context_docs: List[Document]) -> str:
        instruction = (
            "You are a helpful AI assistant. Use the following context to answer the question.\n"
            "If the context doesn't contain enough information to answer the question, say so.\n\n"
            "Context:\n"
        )
        question_part = f"\n\nQuestion: {question}\n\nAnswer:"

        instr_tokens = self.tokenizer.encode(instruction, add_special_tokens=False)
        question_tokens = self.tokenizer.encode(question_part, add_special_tokens=False)
        available_tokens = self.max_length - len(instr_tokens) - len(question_tokens)
        if available_tokens <= 0:
            print("⚠️ Not enough space for context in the prompt!")
            return instruction + "\n\nQuestion: " + question + "\n\nAnswer:"

        # Start with all context docs
        context = "\n\n".join([doc.page_content for doc in context_docs])
        context_tokens = self.tokenizer.encode(context, add_special_tokens=False)

        # Truncate context tokens if needed
        if len(context_tokens) > available_tokens:
            print("⚠️ Context was truncated to fit the model's max token length.")
            context_tokens = context_tokens[:available_tokens]

        truncated_context = self.tokenizer.decode(context_tokens, skip_special_tokens=True)
        prompt = instruction + truncated_context + question_part

        # Final check: re-encode the whole prompt and trim if needed
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        while len(prompt_tokens) > self.max_length:
            # Remove 10 tokens at a time from the context until it fits
            context_tokens = context_tokens[:-10]
            truncated_context = self.tokenizer.decode(context_tokens, skip_special_tokens=True)
            prompt = instruction + truncated_context + question_part
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            print(f"⚠️ Final prompt still too long ({len(prompt_tokens)} tokens), trimming further.")
            if len(context_tokens) == 0:
                print("⚠️ All context removed, but prompt still too long.")
                break

        return prompt
    
    def answer_question_with_context(self, question: str, context_docs: List[Document]) -> str:
        """Answer a question using provided context documents."""
        try:
            # Create RAG prompt with context-aware truncation
            prompt = self.safe_rag_prompt(question, context_docs)
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
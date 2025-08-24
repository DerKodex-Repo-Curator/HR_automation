from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import torch
import os
import json
import shutil
import logging
from datetime import datetime
import logging
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_groq_client(model="llama-3.3-70b-versatile"):
    """Initialize Groq client with proper error handling"""
    if not GROQ_API_KEY:
        raise ValueError("Groq API key not found")
    return ChatGroq(
        api_key=GROQ_API_KEY,
        temperature=0,
        model=model
    )

class DocumentRequest(BaseModel):
    """Structure for document requests"""
    query: str
    employee_id: str
    department: Optional[str] = None
    document_types: Optional[List[str]] = None  # e.g., ["policy", "handbook", "forms"]
    priority: Optional[str] = "normal"  # normal, urgent

class DocumentResponse(BaseModel):
    """Structure for document delivery responses"""
    documents: List[Dict[str, str]] = Field(default_factory=list)  # List of {path: str, title: str, type: str}
    relevance_scores: List[float] = Field(default_factory=list)
    delivery_status: str = "pending"
    delivery_timestamp: Optional[str] = None

class DocumentDeliveryAgent:
    def __init__(
        self,
        documents_path: str = "./company_documents",
        vector_db_path: str = "./doc_vector_store",
        model_name: str = "llama-3.3-70b-versatile",
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        delivery_path: str = "./delivered_documents",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.documents_path = documents_path
        self.vector_db_path = vector_db_path
        self.delivery_path = delivery_path
        
        # Ensure directories exist
        os.makedirs(documents_path, exist_ok=True)
        os.makedirs(vector_db_path, exist_ok=True)
        os.makedirs(delivery_path, exist_ok=True)
        
        # Initialize components
        self._setup_logging()
        self._initialize_embeddings(embedding_model_name)
        self._initialize_llm(model_name)
        self.vector_store = self._initialize_vector_store()
        self._initialize_prompts()

    def _setup_logging(self):
        """Set up logging configuration"""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def _initialize_embeddings(self, embedding_model_name: str):
        """Initialize the embedding model"""
        self.logger.info(f"Initializing embeddings with model: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'device': self.device, 'batch_size': 32}
        )

    def _initialize_llm(self, model_name: str):
        """Initialize the LLM"""
        self.logger.info(f"Initializing Groq with model: {model_name}")
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("Groq API key not found")
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
            model=model_name
        )

    def _initialize_vector_store(self) -> Chroma:
        """Initialize or load the vector store"""
        if os.path.exists(os.path.join(self.vector_db_path, "chroma.sqlite3")):
            self.logger.info("Loading existing Chroma database...")
            return Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
        
        self.logger.info("Creating new Chroma database...")
        documents = self._load_documents()
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.vector_db_path
        )
        vector_store.persist()
        return vector_store

    def _load_documents(self):
        """Load all documents from the documents directory"""
        self.logger.info(f"Loading documents from {self.documents_path}")
        loader = DirectoryLoader(
            self.documents_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        return text_splitter.split_documents(documents)

    def _initialize_prompts(self):
        """Initialize prompt templates"""
        self.document_selection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a document delivery assistant. Help select the most relevant 
            documents based on the employee's request and department context."""),
            ("human", """Request: {query}
            Department: {department}
            Document Types Requested: {document_types}
            
            Please analyze the relevance of the following documents and explain their importance:
            
            {document_summaries}""")
        ])

    def process_document_request(self, request: DocumentRequest) -> DocumentResponse:
        """Process a document request and return relevant documents"""
        try:
            # Search for relevant documents
            search_results = self.vector_store.similarity_search_with_score(
                request.query,
                k=5  # Get top 5 most relevant documents
            )
            
            # Prepare response
            documents = []
            relevance_scores = []
            
            for doc, score in search_results:
                # Filter by document type if specified
                doc_type = self._get_document_type(doc.metadata.get('source', ''))
                if request.document_types and doc_type not in request.document_types:
                    continue
                
                # Prepare document for delivery
                delivered_path = self._prepare_document_delivery(
                    doc.metadata.get('source', ''),
                    request.employee_id
                )
                
                if delivered_path:
                    documents.append({
                        'path': delivered_path,
                        'title': os.path.basename(doc.metadata.get('source', '')),
                        'type': doc_type
                    })
                    relevance_scores.append(score)
            
            # Create response
            response = DocumentResponse(
                documents=documents,
                relevance_scores=relevance_scores,
                delivery_status="completed",
                delivery_timestamp=datetime.now().isoformat()
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing document request: {str(e)}")
            return DocumentResponse(
                delivery_status="failed",
                delivery_timestamp=datetime.now().isoformat()
            )

    def _get_document_type(self, filepath: str) -> str:
        """Determine document type from filepath or content"""
        filepath = filepath.lower()
        if 'policy' in filepath:
            return 'policy'
        elif 'handbook' in filepath:
            return 'handbook'
        elif 'form' in filepath:
            return 'form'
        return 'general'

    def _prepare_document_delivery(self, source_path: str, employee_id: str) -> Optional[str]:
        """Prepare document for delivery by copying to employee's delivery folder"""
        try:
            # Create employee-specific delivery folder
            employee_folder = os.path.join(self.delivery_path, employee_id)
            os.makedirs(employee_folder, exist_ok=True)
            
            # Copy document to delivery folder
            filename = os.path.basename(source_path)
            destination = os.path.join(employee_folder, filename)
            shutil.copy2(source_path, destination)
            
            return destination
        except Exception as e:
            self.logger.error(f"Error preparing document delivery: {str(e)}")
            return None

    def add_new_document(self, filepath: str, document_type: str = None) -> bool:
        """Add a new document to the knowledge base"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Document not found: {filepath}")
                
            # Copy document to documents directory
            filename = os.path.basename(filepath)
            destination = os.path.join(self.documents_path, filename)
            shutil.copy2(filepath, destination)
            
            # Load and process new document
            loader = PyPDFLoader(destination)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            split_docs = text_splitter.split_documents(documents)
            self.vector_store.add_documents(split_docs)
            self.vector_store.persist()
            
            self.logger.info(f"Successfully added new document: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding new document: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize the Document Delivery Agent
    doc_agent = DocumentDeliveryAgent()
    
    # Example request
    request = DocumentRequest(
        query="I need all onboarding documents related to benefits and insurance",
        employee_id="EMP123",
        department="Engineering",
        document_types=["policy", "forms"]
    )
    
    # Process request
    response = doc_agent.process_document_request(request)
    
    # Print results
    print(f"Delivery Status: {response.delivery_status}")
    print(f"Documents Delivered: {len(response.documents)}")
    for doc, score in zip(response.documents, response.relevance_scores):
        print(f"Document: {doc['title']}, Relevance: {score:.2f}")
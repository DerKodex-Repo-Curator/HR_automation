from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import  Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import torch
import os
import logging
from dotenv import load_dotenv



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

class FAQQuery(BaseModel):
    """Structure for incoming FAQ queries"""
    question: str
    employee_id: Optional[str] = None
    department: Optional[str] = None
    context_needed: bool = True

class FAQResponse(BaseModel):
    """Structure for FAQ responses"""
    answer: str
    confidence_score: float
    source_documents: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)

class FAQAgent:
    def __init__(
        self,
        documents_path: str = "./hr_documents",
        vector_db_path: str = "./vector_store",
        model_name: str = "llama-3.3-70b-versatile",
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 0.7
    ):
        self.device = device
        self.documents_path = documents_path
        self.vector_db_path = vector_db_path
        
        # Initialize LLM using Groq
        logger.info(f"Initializing Groq with model: {model_name}")
        self.llm = get_groq_client(model=model_name)
        
        # Initialize Embeddings
        logger.info(f"Initializing embeddings with model: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': 32}
        )
        
        # Initialize Vector Store
        self.vector_store = self._initialize_vector_store()
        
        # Initialize prompts
        self._initialize_prompts()

    def _initialize_vector_store(self) -> Chroma:
        """Initialize or load the vector store"""
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        if os.path.exists(os.path.join(self.vector_db_path, "chroma.sqlite3")):
            logger.info("Loading existing Chroma database...")
            return Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
        else:
            logger.info("Creating new Chroma database...")
            documents = self._load_documents()
            if not documents:
                logger.warning("No documents found to initialize vector store")
                vector_store = Chroma(
                    persist_directory=self.vector_db_path,
                    embedding_function=self.embeddings
                )
                vector_store.persist()
                return vector_store
                
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.vector_db_path
            )
            vector_store.persist()
            return vector_store

    def _load_documents(self) -> List[Document]:
        """Load and process HR documents"""
        logger.info(f"Loading documents from {self.documents_path}")
        if not os.path.exists(self.documents_path):
            logger.warning(f"Documents path {self.documents_path} does not exist")
            return []
            
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
        """Initialize prompt templates with integrated follow-up questions"""
        system_prompt = """You are an HR FAQ assistant. Answer questions based on the context provided.
        Be accurate, concise, and professional. Structure your response in two parts:
        
        1. First, provide a clear and direct answer to the question
        2. Then, within the same response, suggest 2-3 relevant follow-up questions that would help the employee better understand this topic
        
        Always maintain a professional tone and include relevant policy references when available.
        If the context doesn't contain enough information to answer confidently, acknowledge this and suggest reaching out to HR.
        """
        
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """Context: {context}
    Question: {question}
    Department: {department}

    Please provide a complete response including both the answer and relevant follow-up questions within the same response.""")
        ])

    def answer_query(self, query: FAQQuery) -> FAQResponse:
        """Process an FAQ query and return an integrated response"""
        try:
            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(
                query.question,
                k=3
            )
            
            # Prepare context
            context = "\n".join(doc.page_content for doc in relevant_docs)
            
            # Generate integrated response using LLM
            messages = self.qa_prompt.format_messages(
                context=context,
                question=query.question,
                department=query.department or "General"
            )
            
            # Get complete response
            full_response = self.llm.invoke(messages).content
            
            # Split response into answer and follow-up questions if they exist
            response_parts = full_response.split("\nFollow-up questions:", 1)
            main_answer = response_parts[0].strip()
            
            # Extract follow-up questions if they exist
            follow_up_questions = []
            if len(response_parts) > 1:
                questions_text = response_parts[1]
                follow_up_questions = [
                    q.strip() for q in questions_text.split("\n")
                    if q.strip() and "?" in q
                ][:3]
            
            # Calculate confidence score
            confidence_score = min(
                len(relevant_docs) * 0.3,
                0.95
            )
            
            return FAQResponse(
                answer=main_answer,
                confidence_score=confidence_score,
                source_documents=[doc.metadata.get('source', 'Unknown') for doc in relevant_docs],
                follow_up_questions=follow_up_questions
            )
            
        except Exception as e:
            logger.error(f"Error processing FAQ query: {str(e)}")
            return FAQResponse(
                answer="I apologize, but I encountered an error processing your question. Please try again or contact HR directly.",
                confidence_score=0.0
            )

    def update_knowledge_base(self, new_documents_path: str):
        """Update the vector store with new documents"""
        try:
            loader = DirectoryLoader(
                new_documents_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            new_documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            split_docs = text_splitter.split_documents(new_documents)
            self.vector_store.add_documents(split_docs)
            self.vector_store.persist()
            
            logger.info(f"Successfully added {len(split_docs)} new documents to the knowledge base")
            return True
        except Exception as e:
            logger.error(f"Error updating knowledge base: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize the FAQ Agent
    faq_agent = FAQAgent(
        documents_path="./hr_documents",
        vector_db_path="./vector_store",
        model_name="llama-3.3-70b-versatile"
    )
    
    # Example query
    query = FAQQuery(
        question="What are the company's health insurance benefits?",
        department="Engineering",
        employee_id="EMP123"
    )
    
    # Get response
    response = faq_agent.answer_query(query)
    print(f"Answer: {response.answer}")
    print(f"Confidence Score: {response.confidence_score}")
    print(f"Source Documents: {response.source_documents}")
    print(f"Suggested Follow-up Questions: {response.follow_up_questions}")

from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import torch
import os
from enum import Enum
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

class Language(str, Enum):
    """Supported Languages"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ARABIC = "ar"
    CHINESE = "zh"
    HINDI = "hi"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    JAPANESE = "ja"

class TranslationRequest(BaseModel):
    """Structure for translation requests"""
    text: str
    source_language: Optional[Language] = None
    target_language: Language
    context: Optional[str] = None  # e.g., "onboarding", "technical", "HR document"
    employee_id: Optional[str] = None
    department: Optional[str] = None

class TranslationResponse(BaseModel):
    """Structure for translation responses"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence_score: float = Field(default=0.9, ge=0.0, le=1.0)
    linguistic_notes: Optional[List[str]] = Field(default_factory=list)

class LanguagePreference(BaseModel):
    """Employee Language Preferences"""
    employee_id: str
    primary_language: Language
    secondary_languages: Optional[List[Language]] = None
    preferred_communication_style: Optional[str] = None  # formal, casual, technical

class LanguageTranslatorAgent:
    def __init__(
        self,
        vector_db_path: str = "./translator_vector_store",
        model_name: str = "llama-3.3-70b-versatile",
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.vector_db_path = vector_db_path
        
        # Initialize components
        self._setup_logging()
        self._initialize_embeddings(embedding_model_name)
        self._initialize_llm(model_name)
        self._initialize_prompts()
        
        # Initialize vector store for storing language context and preferences
        self.vector_store = self._initialize_vector_store()
        
        # Initialize language preference storage
        self.language_preferences = {}

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
            temperature=0.3,
            model=model_name
        )

    def _initialize_vector_store(self) -> Chroma:
        """Initialize or load the vector store for language context"""
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        if os.path.exists(os.path.join(self.vector_db_path, "chroma.sqlite3")):
            return Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
        
        # Create new vector store for language contexts
        vector_store = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=self.embeddings
        )
        vector_store.persist()
        return vector_store

    def _initialize_prompts(self):
        """Initialize prompt templates for translation"""
        self.translation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional translator and linguistic expert. 
            Translate the given text while maintaining:
            - Original meaning
            - Appropriate tone for the context
            - Cultural nuances
            - Professional language for workplace communication

            Context: {context}
            Source Language: {source_language}
            Target Language: {target_language}"""),
            ("human", "Text to translate: {text}")
        ])

    def set_language_preference(self, preference: LanguagePreference):
        """Store language preferences for an employee"""
        try:
            self.language_preferences[preference.employee_id] = preference
            self.logger.info(f"Language preference set for employee {preference.employee_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error setting language preference: {str(e)}")
            return False

    def translate_text(self, request: TranslationRequest) -> TranslationResponse:
        """Translate text with advanced linguistic analysis"""
        try:
            # Determine source language if not specified
            if not request.source_language:
                # Here you would ideally use a language detection library
                # For simplicity, defaulting to English
                request.source_language = Language.ENGLISH

            # Prepare translation prompt
            messages = self.translation_prompt.format_messages(
                context=request.context or "General workplace communication",
                source_language=request.source_language.value,
                target_language=request.target_language.value,
                text=request.text
            )

            # Generate translation
            translation_result = self.llm.invoke(messages).content

            # Linguistic analysis and notes
            linguistic_notes = self._analyze_translation(
                request.text, 
                translation_result, 
                request.source_language, 
                request.target_language
            )

            return TranslationResponse(
                original_text=request.text,
                translated_text=translation_result,
                source_language=request.source_language.value,
                target_language=request.target_language.value,
                linguistic_notes=linguistic_notes,
                confidence_score=0.9  # High confidence due to advanced model
            )

        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            return TranslationResponse(
                original_text=request.text,
                translated_text="Translation failed",
                source_language=request.source_language.value if request.source_language else "unknown",
                target_language=request.target_language.value,
                confidence_score=0.0
            )

    def _analyze_translation(
        self, 
        original_text: str, 
        translated_text: str, 
        source_lang: Language, 
        target_lang: Language
    ) -> List[str]:
        """Perform linguistic analysis on the translation"""
        notes = []
        
        # Simple linguistic analysis (you can expand this)
        if len(translated_text) > len(original_text) * 1.5:
            notes.append("Translation is significantly longer than original")
        
        if len(translated_text) < len(original_text) * 0.5:
            notes.append("Translation is significantly shorter than original")
        
        return notes

# Example usage
if __name__ == "__main__":
    # Initialize the Language Translator Agent
    translator_agent = LanguageTranslatorAgent()
    
    # Set language preference for an employee
    preference = LanguagePreference(
        employee_id="EMP123",
        primary_language=Language.SPANISH,
        secondary_languages=[Language.ENGLISH],
        preferred_communication_style="professional"
    )
    translator_agent.set_language_preference(preference)
    
    # Example translation request
    translation_request = TranslationRequest(
        text="Welcome to our company! We're excited to have you join our team.",
        source_language=Language.ENGLISH,
        target_language=Language.SPANISH,
        context="Onboarding",
        employee_id="EMP123"
    )
    
    # Translate
    response = translator_agent.translate_text(translation_request)
    
    # Print results
    print(f"Original Text: {response.original_text}")
    print(f"Translated Text: {response.translated_text}")
    print(f"Confidence Score: {response.confidence_score}")
    print(f"Linguistic Notes: {response.linguistic_notes}")

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Union
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import torch
import os
import uuid
import json
from enum import Enum
import logging
from dotenv import load_dotenv
from datetime import datetime

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

class FeedbackSubmissionResponse(BaseModel):
    """Structure for feedback submission response"""
    success: bool
    message: str
    submission_id: str
    timestamp: datetime = Field(default_factory=datetime.now)  


class FeedbackCategory(str, Enum):
    """Predefined feedback categories"""
    OVERALL_EXPERIENCE = "overall_experience"
    ONBOARDING_PROCESS = "onboarding_process"
    TRAINING = "training"
    MANAGER_INTERACTION = "manager_interaction"
    TEAM_INTEGRATION = "team_integration"
    RESOURCES_AND_TOOLS = "resources_and_tools"

class SentimentType(str, Enum):
    """Sentiment classification"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class FeedbackQuestion(BaseModel):
    """Structure for feedback questions"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: FeedbackCategory
    text: str
    question_type: str = "likert"  # likert, open-ended, multiple-choice
    options: Optional[List[str]] = None

class FeedbackSubmission(BaseModel):
    """Structure for feedback submissions"""
    employee_id: str
    department: str
    onboarding_batch: str
    responses: Dict[str, Union[int, str]]
    submitted_at: datetime = Field(default_factory=datetime.now)

    @field_validator('responses')
    def validate_responses(cls, responses):
        """Validate that responses match expected format"""
        if not responses:
            raise ValueError("Responses cannot be empty")
        return responses


class FeedbackAnalysis(BaseModel):
    """Comprehensive feedback analysis"""
    overall_sentiment: SentimentType
    category_scores: Dict[FeedbackCategory, float]
    key_insights: List[str]
    improvement_areas: List[str]
    participation_rate: float

class FeedbackCollectionAgent:
    def __init__(
        self,
        vector_db_path: str = "./feedback_vector_store",
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
        
        # Initialize vector store for storing feedback contexts
        self.vector_store = self._initialize_vector_store()
        
        # Initialize feedback tracking
        self.feedback_templates = self._generate_feedback_templates()
        self.submitted_feedback = {}

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
        """Initialize or load the vector store for feedback context"""
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        if os.path.exists(os.path.join(self.vector_db_path, "chroma.sqlite3")):
            return Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
        
        # Create new vector store for feedback contexts
        vector_store = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=self.embeddings
        )
        vector_store.persist()
        return vector_store

    def _initialize_prompts(self):
        """Initialize prompt templates for feedback analysis"""
        self.feedback_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert HR analyst. Analyze the following employee feedback:
            - Provide an overall sentiment assessment
            - Identify key insights and potential improvement areas
            - Offer constructive recommendations
            
            Context: {context}
            Feedback: {feedback}"""),
            ("human", "Perform a comprehensive feedback analysis.")
        ])

    def _generate_feedback_templates(self) -> List[FeedbackQuestion]:
        """Generate standardized feedback questions"""
        return [
            FeedbackQuestion(
                category=FeedbackCategory.OVERALL_EXPERIENCE,
                text="How would you rate your overall onboarding experience?",
                question_type="likert",
                options=["Very Unsatisfactory", "Unsatisfactory", "Neutral", "Satisfactory", "Very Satisfactory"]
            ),
            FeedbackQuestion(
                category=FeedbackCategory.ONBOARDING_PROCESS,
                text="Were the onboarding materials clear and helpful?",
                question_type="likert",
                options=["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
            ),
            FeedbackQuestion(
                category=FeedbackCategory.TRAINING,
                text="Describe the effectiveness of your initial training.",
                question_type="open-ended"
            ),
            FeedbackQuestion(
                category=FeedbackCategory.MANAGER_INTERACTION,
                text="How supportive was your manager during the onboarding process?",
                question_type="likert",
                options=["Not Supportive", "Slightly Supportive", "Moderately Supportive", "Very Supportive", "Extremely Supportive"]
            )
        ]

    def generate_feedback_form(self, employee_id: str, department: str) -> List[FeedbackQuestion]:
        """Generate a personalized feedback form"""
        try:
            # Could add more personalization logic here based on department, etc.
            return self.feedback_templates
        except Exception as e:
            self.logger.error(f"Error generating feedback form: {str(e)}")
            return []

    def submit_feedback(self, submission: FeedbackSubmission) -> FeedbackSubmissionResponse:
        """Process and store feedback submission"""
        try:
            # Validate submission
            if not submission.responses:
                raise ValueError("No feedback responses provided")
            
            # Store feedback
            self.submitted_feedback[submission.employee_id] = submission
            
            # Add to vector store for analysis
            self.vector_store.add_documents([
                Document(
                    page_content=json.dumps(submission.responses),
                    metadata={
                        "employee_id": submission.employee_id,
                        "department": submission.department,
                        "submitted_at": submission.submitted_at.isoformat()
                    }
                )
            ])
            
            self.logger.info(f"Feedback submitted successfully for {submission.employee_id}")
            return FeedbackSubmissionResponse(
                success=True,
                message="Feedback submitted successfully",
                submission_id=str(uuid.uuid4())
                )
        except Exception as e:
            self.logger.error(f"Feedback submission error: {str(e)}")
            return FeedbackSubmissionResponse(
                success=False,
                message=f"Feedback submission error: {str(e)}",
                submission_id=""
                )

    def analyze_feedback(self, department: Optional[str] = None) -> FeedbackAnalysis:
        """Comprehensive feedback analysis"""
        try:
            # Retrieve all feedback submissions
            submissions = list(self.submitted_feedback.values())
            
            if department:
                submissions = [
                    submission for submission in submissions 
                    if submission.department == department
                ]
            
            if not submissions:
                raise ValueError("No feedback submissions available")
            
            # Prepare feedback for LLM analysis
            feedback_text = "\n".join([
                json.dumps(submission.responses) for submission in submissions
            ])
            
            # Use LLM for in-depth analysis
            messages = self.feedback_analysis_prompt.format_messages(
                context=f"Department: {department or 'All Departments'}",
                feedback=feedback_text
            )
            
            analysis_result = self.llm.invoke(messages).content
            
            # Parse LLM analysis (simplified)
            return FeedbackAnalysis(
                overall_sentiment=self._determine_overall_sentiment(submissions),
                category_scores=self._calculate_category_scores(submissions),
                key_insights=self._extract_key_insights(analysis_result),
                improvement_areas=self._extract_improvement_areas(analysis_result),
                participation_rate=len(submissions) / 100  # Assuming 100 total employees
            )
        except Exception as e:
            self.logger.error(f"Feedback analysis error: {str(e)}")
            return FeedbackAnalysis(
                overall_sentiment=SentimentType.NEUTRAL,
                category_scores={},
                key_insights=[],
                improvement_areas=[],
                participation_rate=0.0
            )

    def _determine_overall_sentiment(self, submissions: List[FeedbackSubmission]) -> SentimentType:
        """Determine overall sentiment from submissions"""
        # Simplified sentiment calculation
        positive_count = sum(
            1 for submission in submissions 
            if any(score >= 4 for score in submission.responses.values() if isinstance(score, int))
        )
        total_submissions = len(submissions)
        
        if positive_count / total_submissions > 0.7:
            return SentimentType.POSITIVE
        elif positive_count / total_submissions < 0.3:
            return SentimentType.NEGATIVE
        return SentimentType.NEUTRAL

    def _calculate_category_scores(self, submissions: List[FeedbackSubmission]) -> Dict[FeedbackCategory, float]:
        """Calculate scores for each feedback category"""
        category_totals = {}
        category_counts = {}
        
        for submission in submissions:
            for question in self.feedback_templates:
                if question.id in submission.responses:
                    score = submission.responses[question.id]
                    if isinstance(score, int):
                        category_totals[question.category] = category_totals.get(question.category, 0) + score
                        category_counts[question.category] = category_counts.get(question.category, 0) + 1
        
        return {
            category: total / category_counts.get(category, 1)
            for category, total in category_totals.items()
        }

    def _extract_key_insights(self, analysis_text: str) -> List[str]:
        """Extract key insights from LLM analysis"""
        # Simplified extraction
        return [
            insight.strip() 
            for insight in analysis_text.split('\n') 
            if insight.strip() and len(insight) > 20
        ][:3]

    def _extract_improvement_areas(self, analysis_text: str) -> List[str]:
        """Extract improvement areas from LLM analysis"""
        # Simplified extraction
        return [
            area.strip() 
            for area in analysis_text.split('\n') 
            if area.strip() and len(area) > 20 and any(keyword in area.lower() for keyword in ['improve', 'change', 'enhance'])
        ][:3]

# Example usage
if __name__ == "__main__":
    # Initialize the Feedback Collection Agent
    feedback_agent = FeedbackCollectionAgent()
    
    # Generate feedback form for an employee
    feedback_form = feedback_agent.generate_feedback_form(
        employee_id="EMP123", 
        department="Engineering"
    )
    
    # Simulate feedback submission
    submission = FeedbackSubmission(
        employee_id="EMP123",
        department="Engineering",
        onboarding_batch="2024-Q1",
        responses={
            question.id: 4 if question.question_type == "likert" else "Great experience overall!"
            for question in feedback_form
        }
    )
    
    # Submit feedback
    feedback_agent.submit_feedback(submission)
    
    # Analyze feedback
    analysis = feedback_agent.analyze_feedback(department="Engineering")
    
    # Print results
    print(f"Overall Sentiment: {analysis.overall_sentiment}")
    print(f"Participation Rate: {analysis.participation_rate}")
    print("Key Insights:", analysis.key_insights)
    print("Improvement Areas:", analysis.improvement_areas)

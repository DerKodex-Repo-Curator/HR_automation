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
from enum import Enum
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(str, Enum):
    TRAINING = "training"
    MEETING = "meeting"
    ONBOARDING = "onboarding"
    OTHER = "other"

class TaskPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TaskRequest(BaseModel):
    task_type: TaskType
    title: str
    description: Optional[str] = None
    employee_id: str
    department: Optional[str] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    preferred_dates: Optional[List[str]] = None
    duration_minutes: int = 60
    attendees: Optional[List[str]] = None
    location: Optional[str] = None
    virtual_meeting: bool = True

class TaskResponse(BaseModel):
    task_id: str
    status: str
    scheduled_time: Optional[str] = None
    meeting_link: Optional[str] = None
    calendar_event_id: Optional[str] = None
    attendees_status: Optional[Dict[str, str]] = None
    reminders: List[Dict[str, str]] = Field(default_factory=list)

class TaskSchedulerAgent:
    def __init__(
        self,
        vector_db_path: str = None,
        model_name: str = "llama-3.3-70b-versatile",
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        calendar_credentials_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Get the directory where task_scheduler_agent.py is located
        self.agent_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set default paths relative to agent directory if not provided
        if vector_db_path is None:
            vector_db_path = os.path.join(self.agent_dir, "scheduler_vector_store")
        if calendar_credentials_path is None:
            calendar_credentials_path = os.path.join(self.agent_dir, "credentials", "calendar_credentials.json")
        
        self.device = device
        self.vector_db_path = vector_db_path
        
        self._setup_logging()
        self._initialize_embeddings(embedding_model_name)
        self._initialize_llm(model_name)
        
        # Make calendar service initialization optional
        try:
            self._initialize_calendar_service(calendar_credentials_path)
        except Exception as e:
            self.logger.warning(f"Calendar service initialization failed: {e}")
            self.logger.info("Continuing without calendar service...")
            self.calendar_service = None
        
        self._initialize_prompts()
        self.vector_store = self._initialize_vector_store()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def _initialize_embeddings(self, embedding_model_name: str):
        self.logger.info(f"Initializing embeddings with model: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'device': self.device, 'batch_size': 32}
        )

    def _initialize_llm(self, model_name: str):
        self.logger.info(f"Initializing Groq with model: {model_name}")
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("Groq API key not found")
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
            model=model_name
        )

    def _initialize_calendar_service(self, credentials_path: str):
        try:
            abs_credentials_path = os.path.abspath(credentials_path)
            self.logger.info(f"Attempting to load credentials from: {abs_credentials_path}")
            
            if not os.path.exists(abs_credentials_path):
                raise FileNotFoundError(f"Credentials file not found at {abs_credentials_path}")

            SCOPES = ['https://www.googleapis.com/auth/calendar']
            
            flow = InstalledAppFlow.from_client_secrets_file(
                abs_credentials_path, SCOPES)
            
            token_path = os.path.join(os.path.dirname(abs_credentials_path), 'token.json')
            
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)
                if not creds.valid:
                    creds = flow.run_local_server(port=8080)
                    with open(token_path, 'w') as token:
                        token.write(creds.to_json())
            else:
                creds = flow.run_local_server(port=8080)
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
            
            self.calendar_service = build('calendar', 'v3', credentials=creds)
            self.logger.info("Successfully initialized calendar service")
        
        except Exception as e:
            self.logger.error(f"Detailed error initializing calendar service: {str(e)}")
            self.calendar_service = None
            raise

    def _initialize_vector_store(self) -> Chroma:
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        if os.path.exists(os.path.join(self.vector_db_path, "chroma.sqlite3")):
            return Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
        
        vector_store = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=self.embeddings
        )
        vector_store.persist()
        return vector_store

    def _initialize_prompts(self):
        self.scheduling_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a scheduling assistant. Help find optimal meeting times based on preferences, priorities, and calendar availability."),
            ("human", """Task Type: {task_type}
            Description: {description}
            Preferred Dates: {preferred_dates}
            Duration: {duration} minutes
            Priority: {priority}
            
            Current Calendar Status:
            {calendar_status}
            
            Please suggest the best scheduling options considering these factors.""")
        ])

    def _find_available_slots(
        self,
        start_date: datetime,
        end_date: datetime,
        duration_minutes: int,
        attendees: List[str]
    ) -> List[Dict[str, datetime]]:
        try:
            available_slots = []
            
            body = {
                "timeMin": start_date.isoformat() + 'Z',
                "timeMax": end_date.isoformat() + 'Z',
                "items": [{"id": attendee} for attendee in attendees]
            }
            
            events_result = self.calendar_service.freebusy().query(body=body).execute()
            
            current_date = start_date
            while current_date < end_date:
                day_start = current_date.replace(hour=9, minute=0)
                day_end = current_date.replace(hour=17, minute=0)
                
                current_slot = day_start
                while current_slot + timedelta(minutes=duration_minutes) <= day_end:
                    slot_end = current_slot + timedelta(minutes=duration_minutes)
                    
                    slot_available = True
                    for attendee in attendees:
                        busy_periods = events_result['calendars'].get(attendee, {}).get('busy', [])
                        for period in busy_periods:
                            period_start = datetime.fromisoformat(period['start'].replace('Z', ''))
                            period_end = datetime.fromisoformat(period['end'].replace('Z', ''))
                            if (current_slot < period_end and slot_end > period_start):
                                slot_available = False
                                break
                    
                    if slot_available:
                        available_slots.append({
                            'start': current_slot,
                            'end': slot_end
                        })
                    
                    current_slot += timedelta(minutes=30)
                current_date += timedelta(days=1)
            
            return available_slots
            
        except Exception as e:
            self.logger.error(f"Error finding available slots: {str(e)}")
            return []

    def _create_calendar_event(
        self,
        task_request: TaskRequest,
        selected_slot: Dict[str, datetime]
    ) -> Dict[str, str]:
        try:
            event = {
                'summary': task_request.title,
                'description': task_request.description,
                'start': {
                    'dateTime': selected_slot['start'].isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': selected_slot['end'].isoformat(),
                    'timeZone': 'UTC',
                },
                'attendees': [{'email': attendee} for attendee in task_request.attendees],
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 24 * 60},
                        {'method': 'popup', 'minutes': 30},
                    ],
                },
            }
            
            if task_request.virtual_meeting:
                event['conferenceData'] = {
                    'createRequest': {
                        'requestId': f"{task_request.task_id}-{datetime.now().timestamp()}",
                        'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                    }
                }
            
            event = self.calendar_service.events().insert(
                calendarId='primary',
                body=event,
                conferenceDataVersion=1 if task_request.virtual_meeting else 0
            ).execute()
            
            return {
                'event_id': event['id'],
                'meeting_link': event.get('hangoutLink', None)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating calendar event: {str(e)}")
            return None

    def schedule_task(self, task_request: TaskRequest) -> TaskResponse:
        task_id = f"{task_request.task_type}-{datetime.now().timestamp()}"
        
        if self.calendar_service is None:
        try:
            task_id = f"{task_request.task_type}-{datetime.now().timestamp()}"
            
            # If calendar service is not available, return a mock response for testing
            if self.calendar_service is None:
                return TaskResponse(
                    task_id=task_id,
                    status="mock_scheduled",
                    scheduled_time=datetime.now().isoformat(),
                    meeting_link="http://mock-meeting-link.com",
                    calendar_event_id="mock-event-123",
                    attendees_status={attendee: "pending" for attendee in (task_request.attendees or [])},
                    reminders=[
                        {"type": "email", "time": "24h before"},
                        {"type": "popup", "time": "30m before"}
                    ]
                )
            
        except Exception as e:
            self.logger.error(f"Error scheduling task: {str(e)}")
            return TaskResponse(
                task_id=task_id,
                status="mock_scheduled",
                task_id=task_id if 'task_id' in locals() else str(datetime.now().timestamp()),
                status="failed",
                scheduled_time=None,
                attendees_status={"error": str(e)}
            )

if __name__ == "__main__":
    scheduler_agent = TaskSchedulerAgent()
    task_request = TaskRequest(
        task_type=TaskType.TRAINING,
        title="Onboarding Training Session",
        description="Initial training session for new employee",
        employee_id="EMP123",
        department="Engineering",
        priority=TaskPriority.HIGH,
        duration_minutes=60,
        attendees=["trainer@company.com", "emp123@company.com"],
        virtual_meeting=True
    )
    
    response = scheduler_agent.schedule_task(task_request)
    
    print(f"Task Status: {response.status}")
    if response.scheduled_time:
        print(f"Scheduled Time: {response.scheduled_time}")
        print(f"Meeting Link: {response.meeting_link}")
        print(f"Calendar Event ID: {response.calendar_event_id}")
        print(f"Attendees Status: {response.attendees_status}")




def schedule_task(self, task_request: TaskRequest) -> TaskResponse:
    task_id = f"{task_request.task_type}-{datetime.now().timestamp()}"
    
    try:
        # If calendar service is not available, return a mock response for testing
        if self.calendar_service is None:
            return TaskResponse(
                task_id=task_id,
                status="mock_scheduled",
                scheduled_time=datetime.now().isoformat(),
                meeting_link="http://mock-meeting-link.com",
                calendar_event_id="mock-event-123",
                attendees_status={attendee: "pending" for attendee in (task_request.attendees or [])},
                reminders=[
                    {"type": "email", "time": "24h before"},
                    {"type": "popup", "time": "30m before"}
                ]
            )
        
        # Add your actual calendar service logic here
        # Example: actual scheduling logic would go here
        
    except Exception as e:
        self.logger.error(f"Error scheduling task: {str(e)}")
        return TaskResponse(
            task_id=task_id,
            status="failed",
            scheduled_time=None,
            meeting_link=None,
            calendar_event_id=None,
            attendees_status={"error": str(e)},
            reminders=[]
        )
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from langchain_groq import ChatGroq
import os
import logging
from datetime import datetime
import logging
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from flask import Flask,request, jsonify
from enum import Enum
import uuid

import json
import sys
import os

import sys
import redis
import time
from rq import Queue
from rq.job import Job
from functools import wraps
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import threading
from circuitbreaker import circuit
import backoff
from logging.handlers import RotatingFileHandler
from rq import Queue
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory



# Get the absolute path to the project's root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from agents.faq_agent.faq_agent import FAQAgent
from agents.document_delivery_agent.document_delivery_agent import DocumentDeliveryAgent 
from agents.task_scheduler_agent.task_scheduler_agent import TaskSchedulerAgent  
from agents.language_translator_agent.language_translator_agent import LanguageTranslatorAgent 
from agents.feedback_collection_agent.feedback_collection_agent import FeedbackCollectionAgent


from agents.faq_agent import FAQQuery
from agents.document_delivery_agent import DocumentRequest
from agents.task_scheduler_agent import TaskRequest, TaskType, TaskPriority
from agents.language_translator_agent import TranslationRequest
from agents.feedback_collection_agent import FeedbackSubmission

coordinator = None

app = Flask(__name__)


# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Prometheus metrics
QUERY_COUNTER = Counter('onboarding_queries_total', 'Total number of onboarding queries', ['query_type'])
QUERY_DURATION = Histogram('query_duration_seconds', 'Time spent processing queries', ['query_type'])
AGENT_ERRORS = Counter('agent_errors_total', 'Total number of agent errors', ['agent_type'])
ACTIVE_QUERIES = Gauge('active_queries', 'Number of queries currently being processed')

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
task_queue = Queue(connection=redis_client)

# Enhanced logging setup
def setup_logging():
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_handler = RotatingFileHandler(
        'onboarding_coordinator.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

logger = setup_logging()

# Cache decorator
def cache_response(ttl_seconds=300):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for key: {cache_key}")
                return json.loads(cached_result)
            
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, ttl_seconds, json.dumps(result))
            logger.info(f"Cache miss for key: {cache_key}, stored new result")
            
            return result
        return wrapper
    return decorator

# Circuit breaker configuration
class CircuitBreakerConfig:
    def __init__(self):
        self.failure_threshold = 5
        self.recovery_timeout = 60
        self.expected_exceptions = (Exception,)

circuit_breaker_config = CircuitBreakerConfig()


def get_groq_client(model="llama-3.3-70b-versatile"):
    """Initialize Groq client with proper error handling"""
    if not GROQ_API_KEY:
        raise ValueError("Groq API key not found")
    return ChatGroq(
        api_key=GROQ_API_KEY,
        temperature=0,
        model=model
    )

class FeedbackResponse(BaseModel):
    """Structure for feedback-related responses in the coordinator"""
    success: bool
    message: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class TaskModel(BaseModel):
    description: str
    expected_output: str
    agent: Any  #  define a proper Agent type


class QueryType(str, Enum):
    FAQ = "faq"
    DOCUMENT = "document"
    TASK = "task"
    TRANSLATION = "translation"
    FEEDBACK = "feedback"

class OnboardingQuery(BaseModel):
    """Structure for incoming onboarding queries"""
    query_type: QueryType
    employee_id: str
    department: str
    content: str
    priority: Optional[str] = "normal"
    language: Optional[str] = "en"
    context: Optional[Dict[str, Any]] = None

class CoordinatorResponse(BaseModel):
    """Structure for coordinator responses"""
    query_id: str
    status: str
    response: Dict[str, Any]
    agent_used: str
    timestamp: datetime = Field(default_factory=datetime.now)

class OnboardingCoordinator:
    def __init__(
        self,
        faq_agent: FAQAgent,
        document_agent: DocumentDeliveryAgent,
        task_agent: TaskSchedulerAgent,
        translator_agent: LanguageTranslatorAgent,
        feedback_agent: FeedbackCollectionAgent
    ):
        self.agents = self._initialize_crew_agents()
        self.specialized_agents = {
            "faq": faq_agent,
            "document": document_agent,
            "task": task_agent,
            "translation": translator_agent,
            "feedback": feedback_agent
        }

        # Initialize message history store
        self.message_store = {}
        
        # Initialize agents with memory
        self._initialize_agents_with_memory()

    
    def _create_resilient_agent(self, agent):
        """Wrap agent with circuit breaker and retry logic"""
        @circuit(failure_threshold=circuit_breaker_config.failure_threshold,
                recovery_timeout=circuit_breaker_config.recovery_timeout)
        @backoff.on_exception(backoff.expo,
                            circuit_breaker_config.expected_exceptions,
                            max_tries=3)
        def resilient_execute(method, *args, **kwargs):
            return method(*args, **kwargs)
        
        wrapped_agent = type(f"Resilient{type(agent).__name__}", (), {})()
        for attr_name in dir(agent):
            if not attr_name.startswith('_'):
                attr = getattr(agent, attr_name)
                if callable(attr):
                    setattr(wrapped_agent, attr_name,
                           lambda m=attr: lambda *a, **kw: resilient_execute(m, *a, **kw))
        
        return wrapped_agent

    def _initialize_crew_agents(self):
        """Initialize CrewAI agents for coordination"""
        coordinator = Agent(
            role="Coordinator",
            goal="Efficiently manage and delegate employee onboarding queries",
            backstory="""You are the central coordinator for the employee onboarding system.
            You analyze requests and delegate them to specialized agents while maintaining
            context and ensuring smooth communication between all components.""",
            allow_delegation=True
        )
        
        analyzer = Agent(
            role="Request Analyzer",
            goal="Analyze and categorize incoming requests accurately",
            backstory="""You specialize in understanding the context and requirements
            of employee queries, ensuring they're routed to the right specialized agent.""",
            allow_delegation=False
        )
        
        integrator = Agent(
            role="Response Integrator",
            goal="Combine and format responses from multiple agents when needed",
            backstory="""You excel at combining information from different sources
            into coherent, unified responses that address all aspects of a query.""",
            allow_delegation=False
        )
        
        return {
            "coordinator": coordinator,
            "analyzer": analyzer,
            "integrator": integrator
        }

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create chat history for a session"""
        if session_id not in self.message_store:
            self.message_store[session_id] = ChatMessageHistory()
        return self.message_store[session_id]
    
    def _initialize_agents_with_memory(self):
        """Initialize all agents with memory management"""
        for agent_type, agent in self.specialized_agents.items():
            # Wrap each agent's main execution method with memory management
            if hasattr(agent, 'execute') and callable(agent.execute):
                agent_with_memory = RunnableWithMessageHistory(
                    agent.execute,
                    self._get_session_history,
                    input_messages_key="content",
                    history_messages_key="chat_history"
                )
                setattr(agent, 'execute_with_memory', agent_with_memory)

    def _create_session_id(self, query: OnboardingQuery) -> str:
        """Create a unique session ID for each employee-department combination"""
        return f"{query.employee_id}_{query.department}_{query.query_type}"

    def process_query(self, query: OnboardingQuery) -> CoordinatorResponse:
        try:
            # Create session ID for this interaction
            session_id = self._create_session_id(query)
            
            # Get the appropriate agent
            primary_agent = query.query_type.value
            agent = self.specialized_agents[primary_agent]
            
            # Prepare the input for the agent
            agent_input = {
                "content": query.content,
                "employee_id": query.employee_id,
                "department": query.department,
                "context": query.context or {}
            }
            
            # Execute with memory management
            if hasattr(agent, 'execute_with_memory'):
                response = agent.execute_with_memory(
                    agent_input,
                    config={"configurable": {"session_id": session_id}}
                )
            else:
                # Fallback for agents without memory support
                response = self._handle_agent_query(query)
            
            return CoordinatorResponse(
                query_id=str(uuid.uuid4()),
                status="completed",
                response=response,
                agent_used=primary_agent,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return CoordinatorResponse(
                query_id=str(uuid.uuid4()),
                status="failed",
                response={"error": str(e)},
                agent_used="none",
                timestamp=datetime.now()
            )

    def _create_analysis_task(self, query: OnboardingQuery) -> Task:
        """Create a task for analyzing the incoming query"""
        description = f"""Analyze the following query:
        Type: {query.query_type}
        Content: {query.content}
        Department: {query.department}
        
        Determine:
        1. Primary agent needed
        2. Additional context required
        3. Need for multi-agent collaboration"""
        
        expected_output = """
        {
            "primary_agent": "string indicating which agent should handle this",
            "additional_context": "dictionary of any additional context needed",
            "needs_integration": "boolean indicating if multiple agents are needed"
        }
        """
        
        return Task(
            description=description,
            agent=self.agents["analyzer"],
            expected_output=expected_output
        )

    def _create_execution_task(self, query: OnboardingQuery, analysis_result: dict) -> Task:
        """Create task for executing the query based on analysis"""
        description = f"""Execute the query using appropriate specialized agent(s):
        Query: {query.content}
        Primary Agent: {analysis_result['primary_agent']}
        Additional Context: {analysis_result['additional_context']}
        
        Coordinate with specialized agents and gather responses."""
        
        expected_output = """
        {
            "response": "dictionary containing the response data",
            "status": "string indicating the execution status"
        }
        """
        
        return Task(
            description=description,
            agent=self.agents["coordinator"],
            expected_output=expected_output
        )

    def _create_integration_task(self, responses: List[Dict[str, Any]]) -> Task:
        """Create task for integrating multiple responses if needed"""
        description = f"""Integrate the following responses into a coherent result:
        Responses: {responses}
        
        Ensure all aspects of the original query are addressed."""
        
        expected_output = """
        {
            "integrated_response": "dictionary containing the combined response",
            "metadata": "dictionary containing integration metadata"
        }
        """
        
        return Task(
            description=description,
            agent=self.agents["integrator"],
            expected_output=expected_output
        )

    def _start_batch_timer(self):
        """Start timer for batch processing"""
        if self._batch_timer is None:
            self._batch_timer = threading.Timer(self.batch_timeout, self._process_current_batch)
            self._batch_timer.start()

    def _process_current_batch(self):
        """Process current batch of queries"""
        with self._batch_lock:
            if not self._current_batch:
                return
            
            batch = self._current_batch
            self._current_batch = []
            self._batch_timer = None
        
        try:
            queries_by_type = {}
            for query in batch:
                query_type = query.query_type
                if query_type not in queries_by_type:
                    queries_by_type[query_type] = []
                queries_by_type[query_type].append(query)
            
            results = []
            for query_type, queries in queries_by_type.items():
                try:
                    if len(queries) == 1:
                        results.append(self._process_single_query(queries[0]))
                    else:
                        results.extend(self._batch_process_queries(queries))
                except Exception as e:
                    logger.error(f"Error processing batch for type {query_type}: {str(e)}")
                    for query in queries:
                        results.append(CoordinatorResponse(
                            query_id=str(uuid.uuid4()),
                            status="failed",
                            response={"error": str(e)},
                            agent_used=query_type
                        ))
            
            return results
        
        finally:
            ACTIVE_QUERIES.dec(len(batch))

    @cache_response(ttl_seconds=300)
    def _process_single_query(self, query: OnboardingQuery) -> CoordinatorResponse:
        """Process a single query with monitoring and error handling"""
        query_type = query.query_type
        QUERY_COUNTER.labels(query_type=query_type).inc()
        ACTIVE_QUERIES.inc()
        
        start_time = time.time()
        try:
            response = self._execute_query(query)
            QUERY_DURATION.labels(query_type=query_type).observe(time.time() - start_time)
            return response
        
        except Exception as e:
            AGENT_ERRORS.labels(agent_type=query_type).inc()
            logger.error(f"Error processing query {query}: {str(e)}", exc_info=True)
            raise
        
        finally:
            ACTIVE_QUERIES.dec()

    def _execute_query(self, query: OnboardingQuery) -> CoordinatorResponse:
        """Execute a single query using the appropriate agent"""
        try:
            primary_agent = query.query_type.value
            
            if primary_agent == "faq":
                response = self._handle_faq_query(query)
            elif primary_agent == "document":
                response = self._handle_document_query(query)
            elif primary_agent == "task":
                response = self._handle_task_query(query)
            elif primary_agent == "translation":
                response = self._handle_translation_query(query)
            elif primary_agent == "feedback":
                response = self._handle_feedback_query(query)
            else:
                raise ValueError(f"Unknown agent type: {primary_agent}")
            
            return CoordinatorResponse(
                query_id=str(uuid.uuid4()),
                status="completed",
                response=response,
                agent_used=primary_agent
            )
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise

    def _batch_process_queries(self, queries: List[OnboardingQuery]) -> List[CoordinatorResponse]:
        """Process a batch of queries of the same type"""
        query_type = queries[0].query_type
        agent = self.specialized_agents[query_type]
        
        try:
            if query_type == "faq":
                return self._batch_process_faq(queries, agent)
            elif query_type == "document":
                return self._batch_process_documents(queries, agent)
            
            return [self._process_single_query(query) for query in queries]
            
        except Exception as e:
            logger.error(f"Batch processing failed for {query_type}: {str(e)}", exc_info=True)
            raise

    def add_to_batch(self, query: OnboardingQuery) -> str:
        """Add query to current batch and return job ID"""
        with self._batch_lock:
            job_id = str(uuid.uuid4())
            self._current_batch.append(query)
            
            if len(self._current_batch) >= self.batch_size:
                task_queue.enqueue(self._process_current_batch)
            else:
                self._start_batch_timer()
            
            return job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a batch processing job"""
        job = Job.fetch(job_id, connection=redis_client)
        return {
            "status": job.get_status(),
            "result": job.result if job.is_finished else None,
            "error": str(job.exc_info) if job.is_failed else None
        }

    @cache_response(ttl_seconds=3600)
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about agent performance"""
        stats = {}
        for agent_type in self.specialized_agents.keys():
            stats[agent_type] = {
                "queries_processed": QUERY_COUNTER.labels(query_type=agent_type)._value.get(),
                "error_count": AGENT_ERRORS.labels(agent_type=agent_type)._value.get(),
                "average_duration": QUERY_DURATION.labels(query_type=agent_type)._sum.get() / 
                                  max(QUERY_DURATION.labels(query_type=agent_type)._count.get(), 1)
            }
        return stats

    def get_conversation_history(self, employee_id: str, department: str, query_type: str) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a specific context"""
        session_id = f"{employee_id}_{department}_{query_type}"
        history = self._get_session_history(session_id)
        
        return [
            {
                "role": "human" if isinstance(msg, HumanMessage) else "ai",
                "content": msg.content,
                "timestamp": msg.additional_kwargs.get("timestamp", datetime.now().isoformat())
            }
            for msg in history.messages
        ]

    def clear_conversation_history(self, employee_id: str, department: str, query_type: str) -> bool:
        """Clear conversation history for a specific context"""
        session_id = f"{employee_id}_{department}_{query_type}"
        if session_id in self.message_store:
            self.message_store[session_id] = ChatMessageHistory()
            return True
        return False



    def process_query(self, query: OnboardingQuery) -> CoordinatorResponse:
        try:
            # Directly determine primary agent from query type
            primary_agent = query.query_type.value
            
            # Execute with specialized agent
            if primary_agent == "faq":
                response = self._handle_faq_query(query)
            elif primary_agent == "document":
                response = self._handle_document_query(query)
            elif primary_agent == "task":
                response = self._handle_task_query(query)
            elif primary_agent == "translation":
                response = self._handle_translation_query(query)
            elif primary_agent == "feedback":
                response = self._handle_feedback_query(query)
            else:
                raise ValueError(f"Unknown agent type: {primary_agent}")
            
            return CoordinatorResponse(
                query_id=str(uuid.uuid4()),
                status="completed",
                response=response,
                agent_used=primary_agent
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return CoordinatorResponse(
                query_id=str(uuid.uuid4()),
                status="failed",
                response={"error": str(e)},
                agent_used="none"
            )

    def _handle_faq_query(self, query: OnboardingQuery) -> Dict[str, Any]:
        """Handle FAQ queries"""
        faq_request = FAQQuery(
            question=query.content,
            employee_id=query.employee_id,
            department=query.department,
            context_needed=True
        )
        return self.specialized_agents["faq"].answer_query(faq_request).dict()

    def _handle_document_query(self, query: OnboardingQuery) -> Dict[str, Any]:
        """Handle document delivery queries"""
        doc_request = DocumentRequest(
            query=query.content,
            employee_id=query.employee_id,
            department=query.department,
            priority=query.priority
        )
        return self.specialized_agents["document"].process_document_request(doc_request).dict()

    def _handle_task_query(self, query: OnboardingQuery) -> Dict[str, Any]:
        """Handle task scheduling queries"""
        task_request = TaskRequest(
            task_type=self._determine_task_type(query.content),
            title=query.content,
            employee_id=query.employee_id,
            department=query.department,
            priority=self._convert_priority(query.priority)
        )
        return self.specialized_agents["task"].schedule_task(task_request).dict()

    def _handle_translation_query(self, query: OnboardingQuery) -> Dict[str, Any]:
        """Handle translation queries"""
        translation_request = TranslationRequest(
            text=query.content,
            target_language=query.language,
            context="onboarding",
            employee_id=query.employee_id,
            department=query.department
        )
        return self.specialized_agents["translation"].translate_text(translation_request).dict()
    
    def _handle_feedback_query(self, query: OnboardingQuery) -> Dict[str, Any]:
        """Handle feedback collection queries"""
        try:
            if "submit" in query.content.lower():
                # Handle feedback submission
                feedback = self._parse_feedback_submission(query)
                response = self.specialized_agents["feedback"].submit_feedback(feedback)
                
                return FeedbackResponse(
                    success=response.success,
                    message=response.message,
                    data={
                        "submission_id": response.submission_id,
                        "employee_id": feedback.employee_id,
                        "department": feedback.department,
                        "timestamp": response.timestamp
                    }
                ).dict()
            else:
                # Handle feedback form generation
                form = self.specialized_agents["feedback"].generate_feedback_form(
                    query.employee_id,
                    query.department
                )
                
                return FeedbackResponse(
                    success=True,
                    message="Feedback form generated successfully",
                    data={
                        "form": [question.dict() for question in form],
                        "employee_id": query.employee_id,
                        "department": query.department
                    }
                ).dict()
                
        except Exception as e:
            logger.error(f"Error handling feedback query: {str(e)}")
            return FeedbackResponse(
                success=False,
                message=f"Error processing feedback query: {str(e)}",
                data={}
            ).dict()


    def _integrate_responses(self, primary_response: Dict[str, Any], 
                           additional_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Integrate multiple responses into a cohesive result"""
        integrated_response = primary_response.copy()
        
        # Add additional context and information
        integrated_response["additional_information"] = []
        for response in additional_responses:
            integrated_response["additional_information"].append({
                "type": response.get("type", "unknown"),
                "content": response.get("content", {}),
                "relevance": response.get("relevance", 1.0)
            })
        
        return integrated_response

    def _determine_task_type(self, content: str) -> TaskType:
        """Determine task type from query content"""
        content = content.lower()
        if "training" in content:
            return TaskType.TRAINING
        elif "meeting" in content:
            return TaskType.MEETING
        elif "onboarding" in content:
            return TaskType.ONBOARDING
        return TaskType.OTHER

    def _convert_priority(self, priority: str) -> TaskPriority:
        """Convert general priority to task priority"""
        priority = priority.lower()
        if priority == "high":
            return TaskPriority.HIGH
        elif priority == "low":
            return TaskPriority.LOW
        return TaskPriority.MEDIUM

    def _parse_feedback_submission(self, query: OnboardingQuery) -> FeedbackSubmission:
        """Parse feedback submission from query content"""
        try:
            # First try to get data from context
            submission_data = query.context if isinstance(query.context, dict) else {}
            
            # If no context, try to parse content as JSON
            if not submission_data and query.content:
                try:
                    submission_data = json.loads(query.content)
                except json.JSONDecodeError:
                    # If content isn't JSON, create a basic feedback structure
                    submission_data = {
                        "responses": {"general_feedback": query.content}
                    }
            
            # Ensure required fields exist
            if "responses" not in submission_data:
                submission_data["responses"] = {}
                
            # Generate batch if not provided
            if "batch" not in submission_data:
                current_date = datetime.now()
                submission_data["batch"] = f"{current_date.year}-Q{(current_date.month-1)//3 + 1}"
            
            return FeedbackSubmission(
                employee_id=query.employee_id,
                department=query.department,
                onboarding_batch=submission_data["batch"],
                responses=submission_data["responses"]
            )
        
        except Exception as e:
            logger.error(f"Error parsing feedback submission: {str(e)}")
            raise ValueError(f"Failed to parse feedback submission: {str(e)}")


# Flask routes

@app.route('/api/onboard/query', methods=['POST'])
def process_query():
    try:
        data = request.get_json()
        query = OnboardingQuery(
            query_type=QueryType(data['query_type']),
            employee_id=data['employee_id'],
            department=data['department'],
            content=data['content'],
            priority=data.get('priority', 'normal'),
            language=data.get('language', 'en'),
            context=data.get('context')
        )
        
        response = coordinator.process_query(query)
        
        return jsonify({
            "query_id": response.query_id,
            "status": response.status,
            "response": response.response,
            "agent_used": response.agent_used,
            "timestamp": response.timestamp.isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": "Error processing query",
            "details": str(e)
        }), 500

@app.route('/api/onboard/types', methods=['GET'])
def get_query_types():
    return jsonify({
        "query_types": [qt.value for qt in QueryType]
    })

# Add new Flask routes for history management
@app.route('/api/onboard/history', methods=['GET'])
def get_history():
    employee_id = request.args.get('employee_id')
    department = request.args.get('department')
    query_type = request.args.get('query_type')
    
    if not all([employee_id, department, query_type]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    history = coordinator.get_conversation_history(employee_id, department, query_type)
    return jsonify({"history": history}), 200

@app.route('/api/onboard/history', methods=['DELETE'])
def clear_history():
    employee_id = request.args.get('employee_id')
    department = request.args.get('department')
    query_type = request.args.get('query_type')
    
    if not all([employee_id, department, query_type]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    success = coordinator.clear_conversation_history(employee_id, department, query_type)
    return jsonify({"success": success}), 200 if success else 404


if __name__ == "__main__":
    # Initialize all specialized agents
    faq_agent = FAQAgent()
    document_agent = DocumentDeliveryAgent()
    task_agent = TaskSchedulerAgent()
    translator_agent = LanguageTranslatorAgent()
    feedback_agent = FeedbackCollectionAgent()
    
    # Initialize coordinator
    coordinator = OnboardingCoordinator(
        faq_agent=faq_agent,
        document_agent=document_agent,
        task_agent=task_agent,
        translator_agent=translator_agent,
        feedback_agent=feedback_agent
    )
    
    # Run the Flask app
    app.run(debug=True)



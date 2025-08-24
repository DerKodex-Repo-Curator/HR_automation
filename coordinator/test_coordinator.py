import requests
import time

def wait_for_server(max_attempts=5, delay=2):
    """Wait for the server to become available"""
    for attempt in range(max_attempts):
        try:
            response = requests.get('http://127.0.0.1:5000/api/onboard/types')
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            print(f"Waiting for server (attempt {attempt + 1}/{max_attempts})...")
            time.sleep(delay)
    return False

def test_query_types():
    """Test the endpoint that returns available query types"""
    try:
        response = requests.get('http://127.0.0.1:5000/api/onboard/types')
        print("\nTesting Query Types Endpoint:")
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Make sure coordinator.py is running.")

def test_faq_query():
    """Test FAQ query processing"""
    print("\nTesting FAQ Query:")
    faq_query = {
        "query_type": "faq",
        "employee_id": "EMP123",
        "department": "Engineering",
        "content": "What is the company's work from home policy?",
        "priority": "normal",
        "language": "en"
    }
    
    try:
        response = requests.post('http://127.0.0.1:5000/api/onboard/query', json=faq_query)
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Make sure coordinator.py is running.")

def test_task_scheduling():
    """Test task scheduling query processing"""
    print("\nTesting Task Scheduling:")
    task_query = {
        "query_type": "task",
        "employee_id": "EMP123",
        "department": "Engineering",
        "content": "Schedule onboarding meeting for tomorrow at 2 PM",
        "priority": "high",
        "language": "en"
    }
    
    try:
        response = requests.post('http://127.0.0.1:5000/api/onboard/query', json=task_query)
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Make sure coordinator.py is running.")

def test_document_request():
    """Test document delivery query processing"""
    print("\nTesting Document Request:")
    doc_query = {
        "query_type": "document",
        "employee_id": "EMP123",
        "department": "Engineering",
        "content": "Please provide the employee handbook",
        "priority": "normal",
        "language": "en"
    }
    
    try:
        response = requests.post('http://127.0.0.1:5000/api/onboard/query', json=doc_query)
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Make sure coordinator.py is running.")

def test_translation_request():
    """Test translation query processing"""
    print("\nTesting Translation Request:")
    translation_query = {
        "query_type": "translation",
        "employee_id": "EMP123",
        "department": "Engineering",
        "content": "Please translate the onboarding documents to Spanish",
        "priority": "normal",
        "language": "es"
    }
    
    try:
        response = requests.post('http://127.0.0.1:5000/api/onboard/query', json=translation_query)
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Make sure coordinator.py is running.")

def test_feedback_submission():
    """Test feedback submission processing"""
    print("\nTesting Feedback Submission:")
    feedback_query = {
        "query_type": "feedback",
        "employee_id": "EMP123",
        "department": "Engineering",
        "content": "Submit feedback: The onboarding process was very helpful",
        "priority": "normal",
        "language": "en"
    }
    
    try:
        response = requests.post('http://127.0.0.1:5000/api/onboard/query', json=feedback_query)
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Make sure coordinator.py is running.")

if __name__ == "__main__":
    print("Starting API tests...")
    print("Make sure coordinator.py is running on http://127.0.0.1:5000")
    
    if wait_for_server():
        # Run all tests
        test_query_types()
        test_faq_query()
        test_task_scheduling()
        test_document_request()
        test_translation_request()
        test_feedback_submission()
    else:
        print("Could not connect to server after multiple attempts. Please check if coordinator.py is running.")

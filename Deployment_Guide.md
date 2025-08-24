# Deployment Guide for Employee Onboarding AI Agent a Multi-Agent System

## Overview

This guide provides step-by-step instructions to deploy the multi-agent system on Windows, macOS, and Linux. This system consists of multiple AI agents working together, including a **coordinator agent** and five specialized agents (**FAQ Agent, Document Delivery Agent, Task Scheduler Agent, Feedback Collection Agent, and Language Translator Agent**). The system utilizes ChromaDB as the vector database for retrieval-augmented generation (RAG) and Redis for logging and caching.

---

## 1. System Requirements

### Hardware Requirements

- Minimum 8GB RAM (Recommended: 16GB+ for larger datasets)
- Minimum 10GB of available disk space
- Multi-core processor (Recommended: Intel i5/i7, AMD Ryzen 5/7, or Apple M1/M2)

### Software Requirements

- **Operating System**: Windows 10/11, macOS 11+ (Big Sur, Monterey, Ventura), or Linux (Ubuntu 20.04+, Debian 11+)
- **Python Version**: 3.10+
- **Virtual Environment**: Recommended for dependency management
- **ChromaDB**: Used as the vector database
- **Redis**: Required for logging and caching
- **Other Dependencies**: Listed in `requirements.txt`

---

## 2. Installation Instructions

### Step 1: Clone or Extract Project Files

Since you will be sending the files directly and not using GitHub, extract the ZIP folder to your preferred directory.

```sh
cd /path/to/extracted/project
```

### Step 2: Create and Activate a Virtual Environment

#### Windows

```sh
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux

```sh
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Run the following command to install all required libraries:

```sh
pip install -r requirements.txt
```

---

## 3. Redis Installation and Setup

Redis is used in the **coordinator agent** for logging and caching.

### Windows (Using WSL - Recommended)

```sh
wsl --install  # If WSL is not installed
wsl
sudo apt-get update
sudo apt-get install redis-server
sudo service redis-server start
```

### Windows (Using Native Installer - Alternative)

1. Download Redis from [Microsoft Archive](https://github.com/microsoftarchive/redis/releases).
2. Install using the `.msi` file.
3. Start Redis manually

```sh
redis-server
```

### macOS

```sh
brew install redis
brew services start redis
```

### Linux (Ubuntu/Debian)

```sh
sudo apt update
sudo apt install redis-server
sudo systemctl enable redis-server --now
```

### Test Redis Installation

```sh
redis-cli ping
# Should respond with "PONG"
```

---

## 4. Configuration

Each agent requires an API key for Groq API, set in a `.env` file inside the respective agent’s folder.

### Example `.env` File (For Each Agent)

```GROQ_API_KEY=your_api_key_here```

Ensure that each agent has a valid `.env` file in its directory before running the system.

---

## 5. Running the Multi-Agent System

### Step 1: Start the Coordinator Agent

The **coordinator agent** is the entry point and manages communication between all agents.
Run the following command inside the `coordinator/` folder:

```sh
python coordinator.py
```

### Step 2: Start Each Specialized Agent (Automatically Initialized)

When `coordinator.py` is run, it will automatically initialize the five specialized agents:

- **FAQ Agent**
- **Document Delivery Agent**
- **Task Scheduler Agent**
- **Feedback Collection Agent**
- **Language Translator Agent**

You do NOT need to start them separately.

---

## 6. Testing the Deployment

Once all agents are running, test the system using API endpoints.

### Example Request (Using `curl` or Postman)

```sh
curl -X POST "http://localhost:5000/api/query" -H "Content-Type: application/json" -d '{"query": "How does the document retrieval system work?"}'
```

Expected response from the **FAQ Agent** (if RAG is used):

```json
{
  "response": "The document retrieval system uses ChromaDB for vector search..."
}
```

---

## 7. Deployment as a Service

To ensure the agents run continuously, set them up as background services.

### Windows (Using Task Scheduler)

1. Open **Task Scheduler**.
2. Create a new task, set trigger **At system startup**.
3. Set action **Start a program** → Browse to Python executable and set arguments to `coordinator.py`.

### macOS/Linux (Using Systemd)

Create a service file in `/etc/systemd/system/agent_service.service`:

```ini
[Unit]
Description=Multi-Agent System
After=network.target

[Service]
User=your_username
WorkingDirectory=/path/to/project
ExecStart=/path/to/project/venv/bin/python /path/to/project/coordinator/coordinator.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```sh
sudo systemctl enable agent_service
sudo systemctl start agent_service
```

Alternatively, run the coordinator in the background using `nohup`:

```sh
nohup python coordinator.py &
```

---

## 8. Troubleshooting & Logs

### Common Issues

- **Missing Dependencies**: Run `pip install -r requirements.txt`.
- **API Key Issues**: Ensure `.env` files are correctly set.
- **Redis Not Running**: Check with `redis-cli ping`.
- **Port Conflicts**: Change the port in `api.py` if another process is using `5000`.

### Viewing Logs

Logs are stored in `logs/` inside each agent’s folder. Use:

```sh
tail -f logs/agent.log
```

---

## 9. Future Enhancements

- **Dockerization**: Containerize the system for easier deployment.
- **Kubernetes**: Scale agents dynamically.
- **Advanced Logging**: Use `loguru` or centralized logging.

---

## 10. Conclusion

This guide outlines the complete deployment process for your multi-agent system on Windows, macOS, and Linux. If any issues arise, refer to the troubleshooting section or reach out for support.

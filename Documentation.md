# Documentation for Employee Onboarding AI Agent a Multi-Agent System

## Overview

This project is a **Employee Onboarding AI Agent a Multi-Agent System** consisting of five specialized agents, each handling a specific task. The system is coordinated by a **central agent**, ensuring seamless interactions between the agents. It also integrates **Retrieval-Augmented Generation (RAG)** for document delivery and FAQs.

## System Architecture

- **Coordinator Agent:** Manages the workflow and communication between agents.
- **FAQ Agent (Uses RAG):** Handles frequently asked questions using document retrieval.
- **Document Delivery Agent (Uses RAG):** Manages document search and retrieval.
- **Task Scheduler Agent:** Organizes and schedules tasks efficiently.
- **Feedback Collection Agent:** Gathers user feedback and stores it for analysis.
- **Language Translator Agent:** Translates content between multiple languages.

## Installation

Supports **Windows, Linux, and macOS**. Follow these steps to install and run the system:

### 1. Clone or Extract the Project

Since the project is shared manually (not via GitHub), extract the ZIP file to your desired location.

### 2. Set Up Dependencies

Install the required dependencies:

```sh
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Each agent folder contains a `.env` file where you need to set your **GROQ_API_KEY**:

```sh
GROQ_API_KEY=your_api_key_here
```

### 4. Run the System

Start the **coordinator agent** to launch the system:

```sh
python coordinator/coordinator.py
```

## Usage

Each agent is designed for a specific function. The **coordinator agent** directs requests to the appropriate agent based on the task type.

## Technologies Used

- **ChromaDB** (Vector Database for RAG-based Agents)
- **Python 3.10+**
- **LangChain for RAG Implementation**
- **Task Scheduling, Feedback Collection, and Translation Modules**

## Notes

- Ensure **Python 3.10+** is installed.
- Only **ChromaDB** is used as a vector database; FAISS is imported but not utilized.
- The project is structured to be easily extensible for future enhancements.

## Future Enhancements

- Add more advanced **retrieval mechanisms**.
- Implement **real-time processing** for feedback and scheduling.
- Extend **language support** for translation tasks.

## Support

For any issues or improvements, modify the respective agent scripts or update the **coordinator agent** for better workflow management.

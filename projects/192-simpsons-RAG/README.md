# Simpsonsbot

A Retrieval-Augmented Generation (RAG) chatbot for answering questions about the Simpsons.

- description of every simpsons episode are scraped from IMDb in ./data
- RAG is implemented using LangChain and ChromaDB
- a backend is implemented using FastAPI
- a frontend is implemented using React

Previous in this series:
- [182-chatbot-react-frontend](../182-chatbot-react-frontend)
- [177-cli-chatbot](../177-cli-chatbox)

![Preview](preview.png)

## Run

Start the backend:
```bash
python app.py
```

Start the frontend:
```bash
npm start
```

## Setup

1. Install Python Dependencies
```bash
pip install -r requirements.txt
```
2. Install Node.js Dependencies
```bash
npm install
```
3. Environment Setup
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```
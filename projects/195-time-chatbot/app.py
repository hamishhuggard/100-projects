from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from dotenv import load_dotenv
from datetime import datetime
import json
import re

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()
GPT_API_KEY = os.getenv("OPENAI_API_KEY")

if not GPT_API_KEY:
    print("Error: OPENAI_API_KEY is not set in the .env file.")
    exit(1)

# Initialize LangChain components
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=GPT_API_KEY,
    temperature=0
)

# Define tools
@tool
def get_time() -> str:
    """Get the current time in HH:MM format."""
    print("get_time tool called")
    return f"Current time: {datetime.now().strftime('%H:%M')}"

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are TimeBot 9000, a helpful AI assistant that always knows the current time. You have a tool to check the current time, and you should always mention that you know the time in your responses."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
tools = [get_time]
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Store conversation history
conversations = {}

def date():
    now = datetime.now()
    return '[' + now.strftime("%Y-%m-%d %H:%M") + ']'

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Initialize or get conversation history
        if session_id not in conversations:
            conversations[session_id] = []
        
        # Add user message to history
        conversations[session_id].append(HumanMessage(content=message))
        
        # Get AI response using LangChain agent
        response = agent_executor.invoke({
            "input": message,
            "chat_history": conversations[session_id][:-1]  # Exclude the current message
        })
        
        gpt_reply = response["output"]
        
        if gpt_reply:
            conversations[session_id].append(AIMessage(content=gpt_reply))
        
        # Log the conversation
        log_file = 'log.chat'
        with open(log_file, 'a') as log:
            log.write(f"{date()} 🙂: {message}\n")
            log.write(f"{date()} 🤖: {gpt_reply}\n\n")
        
        return jsonify({
            'response': gpt_reply,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        if session_id in conversations:
            del conversations[session_id]
        
        return jsonify({'message': 'Conversation reset successfully'})
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/tools', methods=['GET'])
def list_tools():
    """List available tools for debugging"""
    tool_info = [
        {
            "name": tool.name,
            "description": tool.description,
            "args_schema": str(tool.args_schema) if hasattr(tool, 'args_schema') else "No schema"
        }
        for tool in tools
    ]
    return jsonify({'tools': tool_info})

if __name__ == '__main__':
    print("Starting Flask server with LangChain...")
    print("Make sure you have OPENAI_API_KEY set in your .env file")
    print("Available tools:", [tool.name for tool in tools])
    app.run(debug=True, host='0.0.0.0', port=5008) 

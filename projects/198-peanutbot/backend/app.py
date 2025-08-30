from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import json
import re

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()
GPT_API_KEY = os.getenv("OPENAI_API_KEY")

if not GPT_API_KEY:
    print("Error: OPENAI_API_KEY is not set in the .env file.")
    exit(1)

client = OpenAI(api_key=GPT_API_KEY)

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
            conversations[session_id] = [
                {"role": "system", "content": "You are a helpful AI assistant."}
            ]
        
        # Add user message
        conversations[session_id].append({"role": "user", "content": message})
        
        # Get AI response
        completion = client.chat.completions.create(
            model="gpt-5-nano",
            messages=conversations[session_id]
        )
        
        gpt_reply = completion.choices[0].message.content
        
        if gpt_reply:
            conversations[session_id].append({"role": "assistant", "content": gpt_reply})
        
        # Log the conversation
        log_file = 'log.chat'
        with open(log_file, 'a') as log:
            log.write(f"{date()} 🙂 me: {message}\n")
            log.write(f"{date()} 🤖 gpt5: {gpt_reply}\n\n")
        
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

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Make sure you have OPENAI_API_KEY set in your .env file")
    app.run(debug=True, host='0.0.0.0', port=5008) 

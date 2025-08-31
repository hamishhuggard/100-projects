from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from dotenv import load_dotenv
from datetime import datetime
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Global variables for the model
tokenizer = None
model = None

def load_peanutbot_model():
    """Load the trained PeanutBot model"""
    global tokenizer, model
    
    try:
        model_path = "../training/peanutbot-model"
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please run the training script first to generate the model")
            return False
        
        print("Loading PeanutBot model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            "openai-community/gpt2",
            torch_dtype=torch.float32,
            device_map=None
        )
        
        # Load the LoRA weights
        model = PeftModel.from_pretrained(base_model, model_path)
        print("✅ PeanutBot model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def generate_peanut_response(prompt, max_length=100):
    """Generate a response using the PeanutBot model"""
    try:
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=200)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and return response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated text (remove the input prompt)
        if prompt in response:
            response = response[len(prompt):].strip()
        
        return response if response else "🥜 Peanut! 🥜"
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "🥜 Something went peanut-shaped! 🥜"

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
        
        if not model or not tokenizer:
            return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
        
        # Initialize or get conversation history
        if session_id not in conversations:
            conversations[session_id] = []
        
        # Add user message
        conversations[session_id].append({"role": "user", "content": message})
        
        # Create context from conversation history (last few messages)
        context = ""
        for msg in conversations[session_id][-3:]:  # Last 3 messages for context
            if msg["role"] == "user":
                context += f"User: {msg['content']}\n"
            else:
                context += f"PeanutBot: {msg['content']}\n"
        
        # Generate response using PeanutBot
        peanut_reply = generate_peanut_response(context, max_length=150)
        
        if peanut_reply:
            conversations[session_id].append({"role": "assistant", "content": peanut_reply})
        
        # Log the conversation
        log_file = 'log.chat'
        with open(log_file, 'a') as log:
            log.write(f"{date()} 🙂 me: {message}\n")
            log.write(f"{date()} 🥜 peanutbot: {peanut_reply}\n\n")
        
        return jsonify({
            'response': peanut_reply,
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
    model_status = "loaded" if model and tokenizer else "not loaded"
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'model_status': model_status
    })

if __name__ == '__main__':
    print("Starting PeanutBot Flask server...")
    
    # Load the model before starting the server
    if load_peanutbot_model():
        print("🚀 Server starting with PeanutBot model loaded!")
        app.run(debug=True, host='0.0.0.0', port=5008)
    else:
        print("❌ Failed to load model. Server not started.")
        sys.exit(1) 

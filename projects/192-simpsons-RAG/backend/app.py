#!/usr/bin/env python3
"""
Simple Simpsons RAG Chatbot using FastAPI and OpenAI API.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from dotenv import load_dotenv
from datetime import datetime
import json
import re
from pathlib import Path

# ChromaDB imports for RAG
import chromadb
from chromadb.config import Settings

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()
GPT_API_KEY = os.getenv("OPENAI_API_KEY")

if not GPT_API_KEY:
    print("Error: OPENAI_API_KEY is not set in the .env file.")
    exit(1)

# Initialize ChromaDB for RAG
def get_chromadb_client():
    """Get ChromaDB client for episode search"""
    try:
        chroma_path = Path("../data/chroma_db")
        if not chroma_path.exists():
            print("ChromaDB not found. Please run create_db.py first.")
            return None
        
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection("simpsons_episodes")
        return collection
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return None

# Store conversation history
conversations = {}

def date():
    now = datetime.now()
    return '[' + now.strftime("%Y-%m-%d %H:%M") + ']'

def find_relevant_episodes(query: str, limit: int = 3):
    """Find relevant episodes using ChromaDB semantic search"""
    collection = get_chromadb_client()
    if not collection:
        return []
    
    try:
        # Perform semantic search
        results = collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        # Convert results to episode format
        relevant_episodes = []
        if results['metadatas'] and results['metadatas'][0]:
            for metadata in results['metadatas'][0]:
                episode = {
                    'season': metadata.get('season', 'N/A'),
                    'episode_number_in_season': metadata.get('episode_number', 'N/A'),
                    'episode_title': metadata.get('title', 'N/A'),
                    'air_date': metadata.get('air_date', 'N/A'),
                    'description': metadata.get('description', 'N/A'),
                    'imdb_rating': metadata.get('imdb_rating', 'N/A'),
                    'vote_count': metadata.get('vote_count', 'N/A'),
                    'episode_url': metadata.get('url', 'N/A')
                }
                relevant_episodes.append(episode)
        
        return relevant_episodes
        
    except Exception as e:
        print(f"Error in ChromaDB search: {e}")
        return []

def create_context_from_episodes(episodes):
    """Create context string from relevant episodes"""
    if not episodes:
        return "No relevant episodes found."
    
    context_parts = []
    for episode in episodes:
        context_parts.append(f"""
Season {episode.get('season', 'N/A')}, Episode {episode.get('episode_number_in_season', 'N/A')}: {episode.get('episode_title', 'N/A')}
Air Date: {episode.get('air_date', 'N/A')}
Description: {episode.get('description', 'N/A')}
IMDb Rating: {episode.get('imdb_rating', 'N/A')}
""".strip())
    
    return "\n\n---\n\n".join(context_parts)

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
        conversations[session_id].append({"role": "user", "content": message})
        
        # Find relevant episodes using RAG
        relevant_episodes = find_relevant_episodes(message)
        context = create_context_from_episodes(relevant_episodes)
        
        # Create system prompt with context
        system_prompt = f"""You are a helpful AI assistant that answers questions about The Simpsons episodes. 

Use the following episode information to answer the user's question:

{context}

Please provide helpful and accurate answers based on the episode information above. If the question asks about a specific episode, include relevant details like the season/episode number, air date, and description. If you don't find relevant information in the provided episodes, say so clearly.

Keep your answers conversational and engaging, as if you're a big Simpsons fan chatting with a friend."""

        # Create messages for OpenAI
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history (last 5 messages to keep context manageable)
        for msg in conversations[session_id][-5:]:
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            else:
                messages.append({"role": "assistant", "content": msg["content"]})
        
        # Add current user message
        messages.append({"role": "user", "content": message})
        
        # Get response from OpenAI
        from openai import OpenAI
        client = OpenAI(api_key=GPT_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        ai_response = response.choices[0].message.content
        
        # Update conversation history
        conversations[session_id].append({"role": "assistant", "content": ai_response})
        
        # Keep only last 10 messages to prevent context from getting too long
        conversations[session_id] = conversations[session_id][-10:]
        
        # Log the conversation
        log_file = 'log.chat'
        with open(log_file, 'a') as log:
            log.write(f"{date()} 🙂: {message}\n")
            log.write(f"{date()} 🤖: {ai_response}\n\n")
        
        return jsonify({
            'response': ai_response,
            'session_id': session_id,
            'relevant_episodes': relevant_episodes[:1] if relevant_episodes else None
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
    # Check ChromaDB status
    chroma_status = "not_available"
    try:
        collection = get_chromadb_client()
        if collection:
            count = collection.count()
            chroma_status = f"ready ({count} episodes)"
        else:
            chroma_status = "not_found"
    except Exception as e:
        chroma_status = f"error: {str(e)}"
    
    return jsonify({
        'status': 'healthy', 
        'chromadb_status': chroma_status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/episodes', methods=['GET'])
def get_episodes():
    """Get episodes with optional filtering"""
    try:
        collection = get_chromadb_client()
        if not collection:
            return jsonify({'error': 'ChromaDB not available'}), 500
        
        # Get all episodes
        results = collection.get()
        
        episodes = []
        if results['metadatas']:
            for metadata in results['metadatas']:
                episodes.append({
                    'season': metadata.get('season', 'N/A'),
                    'episode_number_in_season': metadata.get('episode_number', 'N/A'),
                    'episode_title': metadata.get('title', 'N/A'),
                    'air_date': metadata.get('air_date', 'N/A'),
                    'description': metadata.get('description', 'N/A'),
                    'imdb_rating': metadata.get('imdb_rating', 'N/A'),
                    'vote_count': metadata.get('vote_count', 'N/A'),
                    'episode_url': metadata.get('url', 'N/A')
                })
        
        # Optional filtering
        season = request.args.get('season', type=int)
        limit = request.args.get('limit', 10, type=int)
        
        if season:
            episodes = [ep for ep in episodes if ep.get('season') == str(season)]
        
        return jsonify({
            "episodes": episodes[:limit],
            "total": len(episodes),
            "returned": min(limit, len(episodes))
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Flask server with Simpsons RAG...")
    print("Make sure you have OPENAI_API_KEY set in your .env file")
    print("Make sure you have run create_db.py to create the ChromaDB")
    app.run(debug=True, host='0.0.0.0', port=5008) 
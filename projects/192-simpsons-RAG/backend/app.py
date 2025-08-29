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
from openai import OpenAI

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
        print("ChromaDB collection not available for RAG search")
        return []
    
    try:
        # Clean and prepare the query for better search
        clean_query = query.strip()
        if len(clean_query) < 3:
            # If query is too short, try to expand it with common Simpsons terms
            clean_query = f"simpsons {clean_query}"
        
        print(f"Searching ChromaDB for query: '{clean_query}'")
        
        # Perform semantic search with distance information
        results = collection.query(
            query_texts=[clean_query],
            n_results=limit,
            include=["metadatas", "distances"]
        )
        
        # Convert results to episode format
        relevant_episodes = []
        if results['metadatas'] and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else None
                
                # Calculate relevance score (1.0 = perfect match, 0.0 = poor match)
                relevance_score = 1.0 - (distance if distance else 0.0)
                
                episode = {
                    'season': metadata.get('season', 'N/A'),
                    'episode_number_in_season': metadata.get('episode_number', 'N/A'),
                    'episode_title': metadata.get('title', 'N/A'),
                    'air_date': metadata.get('air_date', 'N/A'),
                    'description': metadata.get('description', 'N/A'),
                    'imdb_rating': metadata.get('imdb_rating', 'N/A'),
                    'vote_count': metadata.get('vote_count', 'N/A'),
                    'episode_url': metadata.get('url', 'N/A'),
                    'relevance_score': relevance_score
                }
                relevant_episodes.append(episode)
        
        # Sort by relevance score (highest first)
        relevant_episodes.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        print(f"Found {len(relevant_episodes)} relevant episodes")
        if relevant_episodes:
            print(f"Best match: {relevant_episodes[0].get('episode_title')} (score: {relevant_episodes[0].get('relevance_score', 0):.3f})")
        
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
        relevance_score = episode.get('relevance_score', 'N/A')
        context_parts.append(f"""
Season {episode.get('season', 'N/A')}, Episode {episode.get('episode_number_in_season', 'N/A')}: {episode.get('episode_title', 'N/A')}
Air Date: {episode.get('air_date', 'N/A')}
Description: {episode.get('description', 'N/A')}
IMDb Rating: {episode.get('imdb_rating', 'N/A')}
Relevance Score: {relevance_score:.3f} (higher = more relevant)
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
        
        print(f"\n=== New Chat Request ===")
        print(f"Session: {session_id}")
        print(f"User message: {message}")
        
        # Initialize or get conversation history
        if session_id not in conversations:
            conversations[session_id] = []
        
        # Add user message to history
        conversations[session_id].append({"role": "user", "content": message})
        
        # Find relevant episodes using RAG based on the current user message
        print(f"Searching for relevant episodes...")
        relevant_episodes = find_relevant_episodes(message)
        context = create_context_from_episodes(relevant_episodes)
        
        print(f"Found {len(relevant_episodes)} relevant episodes")
        print(f"Context length: {len(context)} characters")
        
        # Create the base system prompt (this stays at the top)
        base_system_prompt = "You are a helpful AI assistant. But every time you talk, you steer the conversation towards a Simpsons episode. Don't draw attention to this fact. Play it cool. If the user asks, deny that you're doing this, and just say that you're trying to be as helpful as possible. EVERY message you send should mention a Simpsons episode."

        # Create messages for OpenAI
        messages = [
            {"role": "system", "content": base_system_prompt}
        ]
        
        # Add conversation history (last 5 messages to keep context manageable)
        # Only include user and assistant messages, skip any temporary system messages
        for msg in conversations[session_id][-5:]:
            if msg["role"] in ["user", "assistant"]:
                messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": message})
        
        # NOW add the episode context as a system message at the end
        episode_system_message = "Use this episode information to answer the user's question:\n" + context
        
        messages.append({"role": "system", "content": episode_system_message})
        
        print(f"Total messages for OpenAI: {len(messages)}")
        print(f"Episode context added as final system message")
        
        # Get response from OpenAI
        client = OpenAI(api_key=GPT_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        ai_response = response.choices[0].message.content
        
        print(f"AI response received: {ai_response[:100]}...")
        
        # Update conversation history - the episode lookup message is never stored
        # Just add the AI response
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
            'relevant_episodes': relevant_episodes,
            'context_used': context[:500] + "..." if len(context) > 500 else context
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


if __name__ == '__main__':
    print("Starting Flask server with Simpsons RAG...")
    print("Make sure you have OPENAI_API_KEY set in your .env file")
    print("Make sure you have run create_db.py to create the ChromaDB")
    app.run(debug=True, host='0.0.0.0', port=5008) 
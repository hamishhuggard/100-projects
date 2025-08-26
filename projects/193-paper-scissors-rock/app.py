from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import json
import re
import random

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()
GPT_API_KEY = os.getenv("OPENAI_API_KEY")

if not GPT_API_KEY:
    print("Error: OPENAI_API_KEY is not set in the .env file.")
    exit(1)

client = OpenAI(api_key=GPT_API_KEY)

# Store conversation history and game states
conversations = {}
game_states = {}

def date():
    now = datetime.now()
    return '[' + now.strftime("%Y-%m-%d %H:%M") + ']'

# Game logic functions
def play_rock_paper_scissors(user_choice):
    choices = ['rock', 'paper', 'scissors']
    bot_choice = random.choice(choices)
    
    if user_choice == bot_choice:
        result = "It's a tie!"
    elif (
        (user_choice == 'rock' and bot_choice == 'scissors') or
        (user_choice == 'paper' and bot_choice == 'rock') or
        (user_choice == 'scissors' and bot_choice == 'paper')
    ):
        result = f"You win! {user_choice.capitalize()} beats {bot_choice}!"
    else:
        result = f"I win! {bot_choice.capitalize()} beats {user_choice}!"
    
    return {
        'user_choice': user_choice,
        'bot_choice': bot_choice,
        'result': result
    }

def play_ultimatum_game(user_choice):
    # Simplified ultimatum game: user proposes split, bot accepts/rejects
    if user_choice == 'fair':  # 50-50 split
        bot_decision = 'accept'
        result = "I accept your fair offer of a 50-50 split!"
    elif user_choice == 'generous':  # 60-40 split in bot's favor
        bot_decision = 'accept'
        result = "I accept your generous offer! Thank you!"
    elif user_choice == 'selfish':  # 80-20 split in user's favor
        bot_decision = 'reject'
        result = "I reject your selfish offer. No deal!"
    else:
        bot_decision = 'reject'
        result = "I don't understand that offer. No deal!"
    
    return {
        'user_offer': user_choice,
        'bot_decision': bot_decision,
        'result': result
    }

def play_prisoners_dilemma(user_choice):
    choices = ['cooperate', 'defect']
    bot_choice = random.choice(choices)
    
    # Payoff matrix
    payoffs = {
        ('cooperate', 'cooperate'): (3, 3),
        ('cooperate', 'defect'): (0, 5),
        ('defect', 'cooperate'): (5, 0),
        ('defect', 'defect'): (1, 1)
    }
    
    user_payoff, bot_payoff = payoffs[(user_choice, bot_choice)]
    
    result = f"You chose to {user_choice}, I chose to {bot_choice}. "
    result += f"Your payoff: {user_payoff}, My payoff: {bot_payoff}"
    
    return {
        'user_choice': user_choice,
        'bot_choice': bot_choice,
        'user_payoff': user_payoff,
        'bot_payoff': bot_payoff,
        'result': result
    }

def play_number_guessing(user_choice):
    # Bot thinks of a number 1-10, user guesses
    try:
        user_guess = int(user_choice)
        bot_number = random.randint(1, 10)
        
        if user_guess == bot_number:
            result = f"Congratulations! You guessed {user_guess} and I was thinking of {bot_number}!"
        elif user_guess < bot_number:
            result = f"Too low! I was thinking of {bot_number}. Your guess was {user_guess}."
        else:
            result = f"Too high! I was thinking of {bot_number}. Your guess was {user_guess}."
        
        return {
            'user_guess': user_guess,
            'bot_number': bot_number,
            'result': result
        }
    except ValueError:
        return {
            'error': 'Please enter a valid number between 1 and 10'
        }

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
                {"role": "system", "content": """You are a friendly AI chatbot that can play games with users. 
                You have access to tools to set up game menus and make game decisions.
                
                Available games:
                - Rock Paper Scissors: Classic hand game
                - Ultimatum Game: User proposes resource split, you accept/reject
                - Prisoner's Dilemma: Classic game theory scenario
                - Number Guessing: You think of a number, user guesses
                
                When a user wants to play a game, use the setup_game_menu tool.
                When they make a choice, use the make_game_decision tool.
                Be enthusiastic about games and explain the rules clearly!"""}
            ]
        
        # Add user message
        conversations[session_id].append({"role": "user", "content": message})
        
        # Check if user wants to play a game
        game_keywords = ['play', 'game', 'rock', 'paper', 'scissors', 'ultimatum', 'prisoner', 'dilemma', 'guess', 'number']
        wants_to_play = any(keyword in message.lower() for keyword in game_keywords)
        
        if wants_to_play:
            # Suggest available games
            response = """I'd love to play a game with you! Here are the games I can play:

🎮 **Rock Paper Scissors** - Classic hand game
💰 **Ultimatum Game** - Propose resource splits
🤝 **Prisoner's Dilemma** - Game theory challenge  
🔢 **Number Guessing** - I think of a number, you guess

Just say which game you'd like to play and I'll set it up for you!"""
        else:
            # Regular chat response
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversations[session_id]
            )
            response = completion.choices[0].message.content
        
        # Add assistant response to conversation
        conversations[session_id].append({"role": "assistant", "content": response})
        
        # Log the conversation
        log_file = 'log.chat'
        with open(log_file, 'a') as log:
            log.write(f"{date()} 🙂 me: {message}\n")
            log.write(f"{date()} 🤖 gpt4: {response}\n\n")
        
        return jsonify({
            'response': response,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/setup_game_menu', methods=['POST'])
def setup_game_menu():
    try:
        data = request.get_json()
        game_type = data.get('game_type', '')
        session_id = data.get('session_id', 'default')
        
        if not game_type:
            return jsonify({'error': 'Game type is required'}), 400
        
        # Initialize game state
        if session_id not in game_states:
            game_states[session_id] = {}
        
        game_states[session_id]['current_game'] = game_type
        game_states[session_id]['game_started'] = True
        
        # Return appropriate menu based on game type
        if game_type == 'rock_paper_scissors':
            menu = {
                'title': 'Rock Paper Scissors',
                'description': 'Choose your weapon!',
                'options': ['rock', 'paper', 'scissors'],
                'instructions': 'Type your choice and press Enter to play!'
            }
        elif game_type == 'ultimatum_game':
            menu = {
                'title': 'Ultimatum Game',
                'description': 'Propose how to split 100 points between us',
                'options': ['fair (50-50)', 'generous (40-60)', 'selfish (80-20)'],
                'instructions': 'Choose your offer strategy and press Enter!'
            }
        elif game_type == 'prisoners_dilemma':
            menu = {
                'title': 'Prisoner\'s Dilemma',
                'description': 'Choose to cooperate or defect',
                'options': ['cooperate', 'defect'],
                'instructions': 'Make your choice and press Enter!'
            }
        elif game_type == 'number_guessing':
            menu = {
                'title': 'Number Guessing',
                'description': 'I\'m thinking of a number between 1 and 10',
                'options': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                'instructions': 'Type your guess and press Enter!'
            }
        else:
            return jsonify({'error': 'Unknown game type'}), 400
        
        return jsonify({
            'success': True,
            'game_menu': menu,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/make_game_decision', methods=['POST'])
def make_game_decision():
    try:
        data = request.get_json()
        user_choice = data.get('user_choice', '')
        session_id = data.get('session_id', 'default')
        
        if not user_choice:
            return jsonify({'error': 'User choice is required'}), 400
        
        if session_id not in game_states or 'current_game' not in game_states[session_id]:
            return jsonify({'error': 'No active game'}), 400
        
        current_game = game_states[session_id]['current_game']
        
        # Play the game based on type
        if current_game == 'rock_paper_scissors':
            result = play_rock_paper_scissors(user_choice.lower())
        elif current_game == 'ultimatum_game':
            result = play_ultimatum_game(user_choice.lower())
        elif current_game == 'prisoners_dilemma':
            result = play_prisoners_dilemma(user_choice.lower())
        elif current_game == 'number_guessing':
            result = play_number_guessing(user_choice)
        else:
            return jsonify({'error': 'Unknown game type'}), 400
        
        # Clear the current game
        game_states[session_id]['current_game'] = None
        game_states[session_id]['game_started'] = False
        
        return jsonify({
            'success': True,
            'game_result': result,
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
        if session_id in game_states:
            del game_states[session_id]
        
        return jsonify({'message': 'Conversation and games reset successfully'})
        
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

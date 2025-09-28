from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import uuid
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'letter-queue-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global letter queue - stores all letters pressed by all users
letter_queue = []

# Color palette for different users
user_colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
]

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle new client connection"""
    user_id = str(uuid.uuid4())[:8]
    color = user_colors[len(letter_queue) % len(user_colors)]
    
    # Send current letter queue to the new client
    emit('initialize', {
        'user_id': user_id,
        'color': color,
        'letter_queue': letter_queue
    })
    
    print(f"Client connected: {user_id}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print("Client disconnected")

@socketio.on('key_pressed')
def handle_key_pressed(data):
    """Handle key press events"""
    letter = data['letter']
    user_id = data['user_id']
    color = data['color']
    timestamp = time.time()
    
    # Add letter to the global queue
    letter_entry = {
        'letter': letter,
        'user_id': user_id,
        'color': color,
        'timestamp': timestamp,
        'id': len(letter_queue)  # Simple ID based on position
    }
    
    letter_queue.append(letter_entry)
    
    # Keep only last 100 letters to prevent memory issues
    if len(letter_queue) > 100:
        letter_queue.pop(0)
    
    # Broadcast the new letter to all clients
    emit('letter_added', {
        'letter': letter,
        'user_id': user_id,
        'color': color,
        'timestamp': timestamp,
        'id': letter_entry['id']
    }, broadcast=True, include_self=False)
    
    print(f"User {user_id} pressed '{letter}' - Queue length: {len(letter_queue)}")


if __name__ == '__main__':
    print("Starting Letter Queue on port 5010")
    print("Open multiple browser windows to test real-time letter tracking")
    socketio.run(app, host='0.0.0.0', port=5008, debug=True)

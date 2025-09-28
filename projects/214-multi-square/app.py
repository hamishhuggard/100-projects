from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'multi-square-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state for all squares - each user gets their own square
squares = {}  # {user_id: {'x': int, 'y': int, 'color': str}}

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
    color = user_colors[len(squares) % len(user_colors)]
    
    # Create a new square for this user
    squares[user_id] = {
        'x': 400,  # Center position
        'y': 300,  # Center position
        'color': color
    }
    
    # Send current state to the new client
    emit('initialize', {
        'user_id': user_id,
        'color': color,
        'all_squares': squares
    })
    
    # Broadcast the new square to all other clients
    emit('square_added', {
        'user_id': user_id,
        'x': squares[user_id]['x'],
        'y': squares[user_id]['y'],
        'color': squares[user_id]['color']
    }, broadcast=True, include_self=False)
    
    print(f"Client connected: {user_id} with color {color}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    # Note: We can't easily identify which user disconnected
    # In a production app, you'd want to track sessions better
    print("Client disconnected")

@socketio.on('square_move')
def handle_square_move(data):
    """Handle square movement"""
    direction = data['direction']
    user_id = data['user_id']
    
    if user_id not in squares:
        return
    
    # Update square position based on direction
    move_distance = 20  # Pixels to move
    
    if direction == 'up':
        squares[user_id]['y'] = max(50, squares[user_id]['y'] - move_distance)
    elif direction == 'down':
        squares[user_id]['y'] = min(550, squares[user_id]['y'] + move_distance)
    elif direction == 'left':
        squares[user_id]['x'] = max(50, squares[user_id]['x'] - move_distance)
    elif direction == 'right':
        squares[user_id]['x'] = min(750, squares[user_id]['x'] + move_distance)
    
    # Broadcast the new coordinates for this specific square to ALL clients
    emit('square_position_update', {
        'user_id': user_id,
        'x': squares[user_id]['x'],
        'y': squares[user_id]['y'],
        'color': squares[user_id]['color']
    }, broadcast=True)
    
    print(f"User {user_id} moved square {direction} to ({squares[user_id]['x']}, {squares[user_id]['y']})")

@socketio.on('reset_position')
def handle_reset_position(data):
    """Reset square to center position"""
    user_id = data['user_id']
    
    if user_id not in squares:
        return
    
    # Reset to center
    squares[user_id]['x'] = 400
    squares[user_id]['y'] = 300
    
    # Broadcast to all clients (including the sender)
    emit('square_position_update', {
        'user_id': user_id,
        'x': squares[user_id]['x'],
        'y': squares[user_id]['y'],
        'color': squares[user_id]['color']
    }, broadcast=True)
    
    print(f"User {user_id} reset square to center")

if __name__ == '__main__':
    print("Starting Multi-Square on port 5012")
    print("Open multiple browser windows to test real-time multi-square movement")
    socketio.run(app, host='0.0.0.0', port=5008, debug=True)

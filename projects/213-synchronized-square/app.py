from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'synchronized-square-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global square position state
square_state = {
    'x': 400,  # Center position (will be updated by frontend)
    'y': 300,  # Center position (will be updated by frontend)
    'color': '#4CAF50'  # Default color
}

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
    color = user_colors[len(square_state) % len(user_colors)]
    
    # Send current square position to the new client
    emit('initialize', {
        'user_id': user_id,
        'color': color,
        'square_state': square_state
    })
    
    print(f"Client connected: {user_id}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print("Client disconnected")

@socketio.on('square_move')
def handle_square_move(data):
    """Handle square movement"""
    direction = data['direction']
    user_id = data['user_id']
    color = data['color']
    
    # Update square position based on direction
    move_distance = 20  # Pixels to move
    
    if direction == 'up':
        square_state['y'] = max(50, square_state['y'] - move_distance)
    elif direction == 'down':
        square_state['y'] = min(550, square_state['y'] + move_distance)
    elif direction == 'left':
        square_state['x'] = max(50, square_state['x'] - move_distance)
    elif direction == 'right':
        square_state['x'] = min(750, square_state['x'] + move_distance)
    
    # Update color to the user who moved it
    square_state['color'] = color
    
    # Broadcast just the movement direction to all clients
    emit('square_moved', {
        'direction': direction,
        'color': square_state['color'],
        'moved_by': user_id
    }, broadcast=True, include_self=False)
    
    print(f"User {user_id} moved square {direction} to ({square_state['x']}, {square_state['y']})")

@socketio.on('reset_position')
def handle_reset_position(data):
    """Reset square to center position"""
    user_id = data['user_id']
    color = data['color']
    
    # Reset to center
    square_state['x'] = 400
    square_state['y'] = 300
    square_state['color'] = color
    
    # Broadcast to all clients
    emit('square_reset', {
        'x': square_state['x'],
        'y': square_state['y'],
        'color': square_state['color'],
        'reset_by': user_id
    }, broadcast=True)
    
    print(f"User {user_id} reset square to center")

if __name__ == '__main__':
    print("Starting Synchronized Square on port 5011")
    print("Open multiple browser windows to test real-time square movement")
    socketio.run(app, host='0.0.0.0', port=5008, debug=True)

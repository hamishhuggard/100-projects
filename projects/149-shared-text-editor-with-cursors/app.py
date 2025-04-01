from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Store document content
value = ''

# Store user cursors - maps user_id to cursor position
cursors = {}

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    # Generate a unique user ID for the new connection
    user_id = str(uuid.uuid4())
    print(f'Client connected: {user_id}')
    
    # Send the user their ID, current document content, and all cursors
    emit('initialize', {
        'user_id': user_id,
        'value': value,
        'cursors': cursors
    })
    
    # Initialize this user's cursor at position 0
    cursors[user_id] = 0
    
    # Notify other users about the new user
    emit('user_joined', {
        'user_id': user_id,
        'cursor_pos': 0
    }, broadcast=True, include_self=False)

@socketio.on('disconnect')
def handle_disconnect():
    # Find and remove the disconnected user's cursor
    for user_id in list(cursors.keys()):
        if request.sid in socketio.server.manager.get_rooms(user_id):
            del cursors[user_id]
            emit('user_left', {'user_id': user_id}, broadcast=True)
            break

@socketio.on('update_value')
def handle_update_value(data):
    global value
    value = data.get('value', '')
    # Broadcast to all clients except sender
    emit('update_value', {'value': value}, broadcast=True, include_self=False)

@socketio.on('update_cursor')
def handle_update_cursor(data):
    user_id = data.get('user_id')
    cursor_pos = data.get('cursor_pos', 0)
    
    if user_id:
        # Update the cursor position for this user
        cursors[user_id] = cursor_pos
        
        # Broadcast the cursor update to all other clients
        emit('cursor_moved', {
            'user_id': user_id,
            'cursor_pos': cursor_pos
        }, broadcast=True, include_self=False)

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5005)
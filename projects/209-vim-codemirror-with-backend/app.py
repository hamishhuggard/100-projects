from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import os
import uuid
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
document_state = {
    'content': '',
    'language': 'javascript',
    'theme': 'default',
    'cursors': {},  # user_id -> cursor position
    'users': {}     # user_id -> user info
}

# File path for persistence
FILE_PATH = 'file.txt'

def load_document():
    """Load document from file.txt if it exists"""
    global document_state
    if os.path.exists(FILE_PATH):
        try:
            with open(FILE_PATH, 'r', encoding='utf-8') as f:
                document_state['content'] = f.read()
        except Exception as e:
            print(f"Error loading file: {e}")
            document_state['content'] = '// Welcome to Collaborative Vim Editor!\n// Start typing to see real-time collaboration\n\nfunction hello() {\n    console.log("Hello, World!");\n}'
    else:
        document_state['content'] = '// Welcome to Collaborative Vim Editor!\n// Start typing to see real-time collaboration\n\nfunction hello() {\n    console.log("Hello, World!");\n}'

def save_document():
    """Save document to file.txt"""
    try:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(document_state['content'])
    except Exception as e:
        print(f"Error saving file: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/document')
def get_document():
    """Get current document state"""
    return jsonify({
        'content': document_state['content'],
        'language': document_state['language'],
        'theme': document_state['theme'],
        'cursors': document_state['cursors'],
        'users': document_state['users']
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    user_id = str(uuid.uuid4())
    user_info = {
        'id': user_id,
        'name': f'User_{user_id[:8]}',
        'color': f'hsl({hash(user_id) % 360}, 70%, 50%)',
        'connected_at': datetime.now().isoformat()
    }
    
    document_state['users'][user_id] = user_info
    document_state['cursors'][user_id] = {'line': 0, 'ch': 0}
    
    # Send current document state to the new client
    emit('document_state', {
        'content': document_state['content'],
        'language': document_state['language'],
        'theme': document_state['theme'],
        'cursors': document_state['cursors'],
        'users': document_state['users']
    })
    
    # Notify other clients about new user
    emit('user_joined', user_info, broadcast=True, include_self=False)
    
    print(f"User {user_id} connected")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    user_id = request.sid
    if user_id in document_state['users']:
        user_info = document_state['users'][user_id]
        del document_state['users'][user_id]
        if user_id in document_state['cursors']:
            del document_state['cursors'][user_id]
        
        # Notify other clients about user leaving
        emit('user_left', {'user_id': user_id}, broadcast=True, include_self=False)
        print(f"User {user_id} disconnected")

@socketio.on('text_change')
def handle_text_change(data):
    """Handle text changes from clients"""
    user_id = request.sid
    
    # Update document content
    document_state['content'] = data['content']
    
    # Save to file
    save_document()
    
    # Broadcast change to all other clients
    emit('text_change', {
        'content': data['content'],
        'user_id': user_id
    }, broadcast=True, include_self=False)

@socketio.on('cursor_change')
def handle_cursor_change(data):
    """Handle cursor position changes"""
    user_id = request.sid
    
    if user_id in document_state['cursors']:
        document_state['cursors'][user_id] = {
            'line': data['line'],
            'ch': data['ch']
        }
        
        # Broadcast cursor change to all other clients
        emit('cursor_change', {
            'user_id': user_id,
            'line': data['line'],
            'ch': data['ch'],
            'user_info': document_state['users'].get(user_id, {})
        }, broadcast=True, include_self=False)

@socketio.on('language_change')
def handle_language_change(data):
    """Handle language mode changes"""
    user_id = request.sid
    document_state['language'] = data['language']
    
    # Broadcast language change to all clients
    emit('language_change', {
        'language': data['language'],
        'user_id': user_id
    }, broadcast=True)

@socketio.on('theme_change')
def handle_theme_change(data):
    """Handle theme changes"""
    user_id = request.sid
    document_state['theme'] = data['theme']
    
    # Broadcast theme change to all clients
    emit('theme_change', {
        'theme': data['theme'],
        'user_id': user_id
    }, broadcast=True)

@socketio.on('vim_mode_change')
def handle_vim_mode_change(data):
    """Handle vim mode changes"""
    user_id = request.sid
    
    # Broadcast vim mode change to all other clients
    emit('vim_mode_change', {
        'user_id': user_id,
        'mode': data['mode'],
        'user_info': document_state['users'].get(user_id, {})
    }, broadcast=True, include_self=False)

@socketio.on('user_info_update')
def handle_user_info_update(data):
    """Handle user info updates (name, etc.)"""
    user_id = request.sid
    
    if user_id in document_state['users']:
        document_state['users'][user_id].update(data)
        
        # Broadcast user info update to all clients
        emit('user_info_update', {
            'user_id': user_id,
            'user_info': document_state['users'][user_id]
        }, broadcast=True)

if __name__ == '__main__':
    # Load initial document
    load_document()
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("Starting Collaborative Vim Editor Server...")
    print("Document loaded from:", FILE_PATH if os.path.exists(FILE_PATH) else "default content")
    print("Server will be available at: http://localhost:5000")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5008)

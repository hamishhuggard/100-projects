from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'synchronized-checkboxes-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state for the 4 checkboxes
checkbox_states = {
    'checkbox1': False,
    'checkbox2': False,
    'checkbox3': False,
    'checkbox4': False
}

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle new client connection"""
    user_id = str(uuid.uuid4())[:8]
    
    # Send current checkbox states to the new client
    emit('initialize', {
        'user_id': user_id,
        'checkbox_states': checkbox_states
    })
    
    print(f"Client connected: {user_id}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print("Client disconnected")

@socketio.on('checkbox_change')
def handle_checkbox_change(data):
    """Handle checkbox state changes"""
    checkbox_id = data['checkbox_id']
    is_checked = data['is_checked']
    user_id = data['user_id']
    
    # Update the global state
    if checkbox_id in checkbox_states:
        checkbox_states[checkbox_id] = is_checked
        
        # Broadcast to all other clients
        emit('checkbox_update', {
            'checkbox_id': checkbox_id,
            'is_checked': is_checked,
            'user_id': user_id
        }, broadcast=True, include_self=False)
        
        print(f"User {user_id} changed {checkbox_id} to {is_checked}")

@socketio.on('reset_all')
def handle_reset_all(data):
    """Reset all checkboxes to unchecked state"""
    user_id = data['user_id']
    
    # Reset all checkboxes
    for checkbox_id in checkbox_states:
        checkbox_states[checkbox_id] = False
    
    # Broadcast to all clients
    emit('all_checkboxes_reset', {
        'user_id': user_id
    }, broadcast=True)
    
    print(f"User {user_id} reset all checkboxes")

if __name__ == '__main__':
    print("Starting Synchronized Checkboxes on port 5009")
    print("Open multiple browser windows to test synchronization")
    socketio.run(app, host='0.0.0.0', port=5008, debug=True)

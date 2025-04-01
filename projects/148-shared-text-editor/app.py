from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Store document content as a single variable
value = ''

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send current content to newly connected client
    emit('update_value', {'content': value})

@socketio.on('update_value')
def handle_update_value(data):
    global value
    value = data.get('value', '')
    # Broadcast to all clients except sender
    emit('update_value', {'value': value}, broadcast=True, include_self=False)

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5005)
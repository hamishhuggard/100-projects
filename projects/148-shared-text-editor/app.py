from flask import Flask, render_template
from flask_socketio import SocketIO, emit, join_room

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Store document content
documents = {
    'default': ''
}

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('join')
def handle_join(data):
    room = data.get('room', 'default')
    join_room(room)
    emit('document_content', {'content': documents.get(room, '')})

@socketio.on('text_change')
def handle_text_change(data):
    room = data.get('room', 'default')
    content = data.get('content', '')
    documents[room] = content
    emit('text_update', {'content': content}, to=room, include_self=False)

if __name__ == '__main__':
    socketio.run(app, debug=True)
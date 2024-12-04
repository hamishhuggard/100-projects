from flask import Flask, jsonify, requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

tasks = []

@app.route('/tasks', methods=['GET'])
def get_tasks():
    return jsonify([task for task in tasks if not task['deleted']])


@app.route('/tasks', methods=['POST'])
def add_task():
    data = request.json
    task = { 'id': len(tasks), 'title': data['title'], 'complete': False, 'deleted': False }
    tasks.append(task)
    return jsonify(task), 201

@app.route('/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    data = request.json
    tasks = tasks[task_id]
    task['complete'] = True
    return jsonify(task), 201

@app.route('/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    tasks[task_id]['deleted'] = True
    return jsonify({'message': 'task_deleted'})

if __name__ == "__main__":
    app.run(debug=True)

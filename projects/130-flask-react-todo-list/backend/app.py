from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

tasks = []

@app.route('/tasks', methods=['GET'])
def get_tasks():
    print(tasks)
    return jsonify([task for task in tasks if not task['deleted']])


@app.route('/tasks', methods=['POST'])
def add_task():
    data = request.json
    task = { 'id': len(tasks), 'title': data['title'], 'complete': False, 'deleted': False }
    tasks.append(task)
    print(tasks)
    return jsonify(task), 201

@app.route('/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    data = request.json
    print(data)
    task = tasks[task_id]
    task['complete'] = request.json['complete']
    print(tasks)
    return jsonify(task), 201

@app.route('/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    tasks[task_id]['deleted'] = True
    print(tasks)
    return jsonify({'message': 'task_deleted'})

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, url_for, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///todo.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task = db.Column(db.String(200), nullable=False)
    completed = db.Column(db.Boolean, default=False)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    todos = Todo.query.all()
    return render_template('index.html', todos=todos)

@app.route('/add', methods=['POST'])
def add():
    task_content = request.form['task']
    new_task = Todo(task=task_content)
    db.session.add(new_task)
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/delete/<int:todo_id>')
def delete(todo_id):
    task = Todo.query.get_or_404(todo_id)
    db.session.delete(task)
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/complete/<int:todo_id>')
def complete(todo_id):
    task = Todo.query.get_or_404(todo_id)
    task.completed = not task.completed
    db.session.commit()
    return redirect(url_for('index'))

if __name__=="__main__":
    app.run(debug=True)

from flask import Flask, redirect, url_for, request, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///todo.db'
db = SQLAlchemy(app)

class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task = db.Column(db.String(200))
    completed = db.Column(db.Boolean, default=False)

with app.app_context():
    db.create_all()

@app.route('/delete/<int:id>')
def delete(id):
    todo = Todo.query.get(id)
    db.session.delete(todo)
    db.session.commit()
    return redirect('index')

@app.route('/complete/<int:id>')
def complete(id):
    todo = Todo.query.get(id)
    todo.completed = not todo.completed
    db.session.commit()
    return redirect('index')

@app.route('/new', methods=['POST'])
def new():
    new_task = Todo(**request.form)
    db.session.add(new_task)
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/')
def index():
    todos = Todo.query.all()
    return render_template('index.html', todos=todos)

if __name__ == "__main__":
    app.run()

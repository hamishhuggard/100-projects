from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///todos.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)

# define database model
class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task = db.Column(db.String(200), nullable=False)
    completed = db.Column(db.Boolean, default=False)

# initialize db
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    # fetch all tasks from db
    todos = Todo.query.all()
    return render_template('index.html', todos=todos)

@app.route('/add', methods=['GET'])
def app():
#    task_content = request.form['task']
#    new_task = Todo(task=task_content)
#    db.session.add(new_task)
#    db.session.commmit()
    return redirect(url_for('index'))
#
#@app.route('/delete/<int:todo_id>')
#def delete(todo_id):
#    todo = Todo.get_or_404(todo_id)
#    db.session.delete(todo)
#    db.sesssion.commit()
#    return redirect(url_for('index'))
#
#@app.route('/complete/<int:todo_id>')
#def complete(todo_id):
#    todo = Todo.get_or_404(todo_id)
#    todo.complete = not todo.complete
#    db.sesssion.commit()
#    return redirect(url_for('index'))
#
if __name__=="__main__":
    app.run(debug=True)




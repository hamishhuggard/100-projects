from flask import Flask, render_template, redirect, request, url_for
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///login.db'
db = SQLAlchemy(app)

class User(db.Model):
    username = db.Column(db.String(10), primary_key=True)
    password = db.Column(db.String(10))

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('signup.html')
    if request.method == 'POST':
        username = request.form["username"]
        password = request.form["password"]
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('user_page', username=username))

@app.route('/login', methods=['GET', 'POST'])
def login():
    attempts = request.args.get('attempts')
    incorrect = request.args.get('attempts')
    if request.method == 'GET':
        if incorrect:
            return render_template('login.html', attempts=attempts, incorrect=incorrect)
        else:
            return render_template('login.html')
    try:
        assert request.method == 'POST'
        username = request.form["username"]
        password = request.form["password"]
        assert username
        user_obj = User.query.filter_by(username=username).first()
        assert user_obj
        assert user_obj.password == password
        return redirect(url_for('user_page', username=username))
    except AssertionError:
        attempts = int(attempts)-1 if attempts else 5
        if attempts < 1:
            return redirect(url_for('locked'))
        return redirect(url_for('login', incorrect=True, attempts=attempts))


@app.route('/user/<string:username>')
def user_page(username):
    return render_template('user_page.html', username=username)

@app.route('/locked')
def locked():
    return render_template('locked.html')

if __name__ == "__main__":
    app.run(debug=True)

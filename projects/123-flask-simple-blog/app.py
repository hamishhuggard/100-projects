from flask import Flask, request, redirect, url_for, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///books.db'
db = SQLAlchemy(app)

class User(db.Model):
    id    = db.Column(db.Integer, primary_key=True)
    posts = db.relationship('Post', backref='author', lazy='joined')

class Post(db.Model):
    id      = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    title   = db.Column(db.String(200))
    content = db.Column(db.String(2000))

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    posts = Post.query.all()
    return render_template('index.html', posts=posts)

@app.route('/user/<string:username>', methods=['GET', 'DELETE'])
@app.route('/post', methods=['POST'])
def new_post():
    new_post = Post(**request.form)
    db.session.add(new_post)
    db.session.commit()
    return redirect(url_for('post', post_id=new_post.id))

@app.route('/post/<int:post_id>', methods=['GET', 'DELETE', 'PUT'])
def post(post_id=None):
    method = request.method
    if post_id:
        post = Post.query.get(post_id)
    if method == 'GET':
        return render_template('post.html', post=post)
    if method == 'DELETE':
        db.session.delete(post)
        db.session.commit()
        return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, flash, session
from datetime import datetime
import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Ensure the instance folder exists
if not os.path.exists('instance'):
    os.makedirs('instance')

# Database setup
def get_db_connection():
    conn = sqlite3.connect('instance/blog.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            author_id INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (author_id) REFERENCES users (id)
        );
    ''')
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# Routes
@app.route('/')
def index():
    conn = get_db_connection()
    posts = conn.execute('''
        SELECT p.id, p.title, p.content, p.created_at, u.username 
        FROM posts p JOIN users u ON p.author_id = u.id
        ORDER BY p.created_at DESC
    ''').fetchall()
    conn.close()
    return render_template('index.html', posts=posts)

@app.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        conn = get_db_connection()
        error = None
        
        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'
        elif not email:
            error = 'Email is required.'
        elif conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone() is not None:
            error = f'User {username} is already registered.'
        
        if error is None:
            conn.execute(
                'INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                (username, generate_password_hash(password), email)
            )
            conn.commit()
            conn.close()
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        
        flash(error)
        conn.close()
    
    return render_template('register.html')

@app.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        error = None
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        
        if user is None:
            error = 'Incorrect username.'
        elif not check_password_hash(user['password'], password):
            error = 'Incorrect password.'
        
        if error is None:
            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            conn.close()
            return redirect(url_for('index'))
        
        flash(error)
        conn.close()
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/create', methods=('GET', 'POST'))
def create():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        
        error = None
        if not title:
            error = 'Title is required.'
        
        if error is not None:
            flash(error)
        else:
            conn = get_db_connection()
            conn.execute(
                'INSERT INTO posts (title, content, author_id) VALUES (?, ?, ?)',
                (title, content, session['user_id'])
            )
            conn.commit()
            conn.close()
            return redirect(url_for('index'))
    
    return render_template('create.html')

@app.route('/post/<int:id>')
def post(id):
    conn = get_db_connection()
    post = conn.execute('''
        SELECT p.id, p.title, p.content, p.created_at, p.author_id, u.username 
        FROM posts p JOIN users u ON p.author_id = u.id
        WHERE p.id = ?
    ''', (id,)).fetchone()
    conn.close()
    
    if post is None:
        return redirect(url_for('index'))
    
    return render_template('post.html', post=post)

@app.route('/edit/<int:id>', methods=('GET', 'POST'))
def edit(id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    post = conn.execute('SELECT * FROM posts WHERE id = ?', (id,)).fetchone()
    
    if post is None:
        conn.close()
        return redirect(url_for('index'))
    
    # Check if the current user is the author
    if post['author_id'] != session['user_id']:
        conn.close()
        flash('You can only edit your own posts.')
        return redirect(url_for('post', id=id))
    
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        
        error = None
        if not title:
            error = 'Title is required.'
        
        if error is not None:
            flash(error)
        else:
            conn.execute(
                'UPDATE posts SET title = ?, content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                (title, content, id)
            )
            conn.commit()
            conn.close()
            return redirect(url_for('post', id=id))
    
    conn.close()
    return render_template('edit.html', post=post)

@app.route('/delete/<int:id>', methods=('POST',))
def delete(id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    post = conn.execute('SELECT * FROM posts WHERE id = ?', (id,)).fetchone()
    
    if post is None:
        conn.close()
        return redirect(url_for('index'))
    
    # Check if the current user is the author
    if post['author_id'] != session['user_id']:
        conn.close()
        flash('You can only delete your own posts.')
        return redirect(url_for('post', id=id))
    
    conn.execute('DELETE FROM posts WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    flash('Post deleted successfully!')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True) 
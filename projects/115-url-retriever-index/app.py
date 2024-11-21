from flask import Flask, request, render_template_string, redirect, url_for
import requests
from bs4 import BeautifulSoup
from readability import Document
import sqlite3
import os

app = Flask(__name__)

# template for viewing all extracted pages
homepage_template = """
<!doctype html>
<html>
<head>
    <title> Webpage Extractor </title>
</head>
<body>
    <h1> Webpage Extractor </h1>
    <form>
        <input type="text" name="url" placeholder="Enter a URL" required>
        <button type="submit"> Extract </button>
    </form>
    <h2> Extracted pages: </h2>
    <ul>
        {% for page in pages %}
            <li>
                <a href="{{ url_for('view_page', page_id=page['id'] }}">{{ page['url'] }}</a>
            </li>
        {% endfor %}
    </ul>
</body>
</html>
"""

# template for viewing a single extracted page
page_template = """
<!doctype html>
<html>
<head>
    <title> Extracted Content </title>
</head>
<body>
    <a href="{{ url_for('index' }}">Back to home</a>
    <h1> Extracted Content </h1>
    <div style="white-space: pre-wrap; border: 1px solid #ccc; padding: 10px;">
        {{ content|safe }}
    </div>
</body>
</html>
"""

# initialize db
def init_db():
    if not os.path.exists("cache.db"):
        conn = sqlite3.connect("cache.db")
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                content TEXT
            );
        ''')
        conn.commit();
        conn.close();

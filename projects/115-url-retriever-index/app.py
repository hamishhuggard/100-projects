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
    <form method="POST" action="/">
        <input type="text" name="url" placeholder="Enter a URL" required>
        <button type="submit"> Extract </button>
    </form>
    <h2> Extracted pages: </h2>
    <ul>
        {% for page in pages %}
            <li>
                <a href="{{ url_for('view_page', page_id=page['id']) }}">{{ page['url'] }}</a>
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
    <a href="{{ url_for('index') }}">Back to home</a>
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

# add a URL to the database
def add_url_to_cache(url):
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()

    # check if url already exists
    cursor.execute("SELECT id from cache WHERE url = ?", (url,))
    if cursor.fetchone():
        conn.close()
        return

    try:
        # fetch and extract content
        response = requests.get(url)
        response.raise_for_status()
        doc = Document(response.text)
        content = BeautifulSoup(doc.summary(), "html.parser").get_text()

        # insert into db
        cursor.execute("INSERT INTO cache (url, content) VALUES (?, ?)", (url, content))
        conn.commit()

    except Exception as e:
        print(f"Error fetching {url}: {e}")
    finally:
        conn.close()


# get all cached pages
def get_all_cached_pages():
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, url FROM cache")
    pages = [{"id": row[0], "url": row[1]} for row in cursor.fetchall()]
    conn.close()
    return pages

# get content of a single page
def get_page_content(page_id):
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    cursor.execute("SELECT url, content FROM cache WHERE id = ?", (page_id,))
    row = cursor.fetchone()
    conn.close()
    return row[1] if row else None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")
        add_url_to_cache(url)
        return redirect(url_for("index"))

    pages = get_all_cached_pages()
    return render_template_string(homepage_template, pages=pages)

@app.route("/page/<int:page_id>/")
def view_page(page_id):
    content = get_page_content(page_id)
    if not content:
        return "Page not found", 404
    return render_template_string(page_template, content=content)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)

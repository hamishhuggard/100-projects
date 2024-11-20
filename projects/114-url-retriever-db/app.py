from flask import Flask, request, render_template_string
import requests
from bs4 import BeautifulSoup
from readability import Document
import sqlite3
import os

app = Flask(__name__)

# html template
html_template = """
<!doctype html>
<html>
<head>
    <title>Text Extractor with Caching</title>
</head>
<body>
    <h1>Enter a URL to extract main content:</h1>
    <form method="POST">
        <input type="text" name="url" placeholder="Enter URL" required>
        <button type="submit">Extract</button>
    </form>
    {% if content %}
        <h2>Extracted Content:</h2>
        <div style="white-space: pre-wrap; border: 1px solid #ccc; padding: 10px;">
            {{ content|safe }}
        </div>
    {% endif %}
</body>
</html>
"""

# initialize database
def init_db():
    if not os.path.exists("cache.db"):
        conn = sqlite3.connect("cache.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                content TEXT
            )
        """)
        conn.commit()
        conn.close()

# fetch content from cache or scrape if not cached
def get_cached_or_fetch_content(url):
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()

    # check if url is cached
    cursor.execute("SELECT content FROM cache WHERE url = ?", (url,))
    row = cursor.fetchone()
    if row:
        conn.close()
        return row[0]  # return cached content

    # fetch and extract content if not cached
    try:
        response = requests.get(url)
        response.raise_for_status()
        doc = Document(response.text)
        content = BeautifulSoup(doc.summary(), "html.parser").get_text()

        # cache the new content
        cursor.execute("INSERT INTO cache (url, content) VALUES (?, ?)", (url, content))
        conn.commit()
    except Exception as e:
        content = f"Error fetching content: {str(e)}"

    conn.close()
    return content

@app.route("/", methods=["GET", "POST"])
def index():
    content = None
    if request.method == "POST":
        url = request.form.get("url")
        content = get_cached_or_fetch_content(url)
    return render_template_string(html_template, content=content)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)

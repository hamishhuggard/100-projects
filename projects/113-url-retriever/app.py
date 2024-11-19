from flask import Flask, request, render_template_string
import requests
from bs4 import BeautifulSoup
from readability import Document

app = Flask(__name__)

# html template
html_template = """
<!doctype html>
<html>
<head>
    <title>Text Extractor</title>
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

@app.route("/", methods=["GET", "POST"])
def index():
    content = None
    if request.method == "POST":
        url = request.form.get("url")
        try:
            response = requests.get(url)
            response.raise_for_status()  # raise error for invalid response
            doc = Document(response.text)
            content = BeautifulSoup(doc.summary(), "html.parser").get_text()
        except Exception as e:
            content = f"Error fetching content: {str(e)}"
    return render_template_string(html_template, content=content)

if __name__ == "__main__":
    app.run(debug=True)

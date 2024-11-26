from flask import redirect, url_for, Flask, render_template
from math import sqrt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', index=list(range(1,100)))

def factorize(n):
    return [ i for i in range(1, n+1) if n%i==0 ]

@app.route('/<int:n>')
def int_page(n):
    return render_template('number.html', n=n, factors=factorize(n))

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, render_template_string, redirect, url_for
import requests
from bs4 import BeautifulSoup
from readability import Document
import sqlite3
import os

app = Flask(__name__)

homepage_html = """
<!doctype html>
<html>
<head>
    <title>
    </title>
</head>
<body>
    <h1> Webpage Extractor </h1>
    <form>
    <input>
    <button>
    </form>
    <h2>
    </h2>
    <ul>
        <li>
        </li>
        <a>
        </a>
    </ul>
</body>
</html>
"""

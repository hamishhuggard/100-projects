import csv
import requests
from bs4 import BeautifulSoup

# Google Sheets CSV URL
csv_url = "https://docs.google.com/spreadsheets/d/124aXpzY9Q3a_9E-q7LqXed7DCUIMlSbPJBa-OBkQgyA/export?format=csv&gid=1383994667&output=csv"

# Download CSV
response = requests.get(csv_url)
csv_data = response.content.decode('utf-8').splitlines()[:20]

# Read CSV and extract words from the "Word" column
words = []
reader = csv.DictReader(csv_data)
for row in reader:
    words.append(row['Word'])

etymonline_list = []
wiktionary_list = []
prompt_list = []
for word in words:
    # Fetch etymonline data
    etymonline_url = f"https://www.etymonline.com/search?q={word}"
    etymonline_page = requests.get(etymonline_url)
    etymonline_soup = BeautifulSoup(etymonline_page.content, 'html.parser')
    etymonline = etymonline_soup.find(class_="word__etymology_expand--1s7tE")
    etymonline_text = etymonline.get_text(strip=True) if etymonline else "No data found"

    # Fetch Wiktionary data
    wiktionary_url = f"https://en.wiktionary.org/wiki/{word}"
    wiktionary_page = requests.get(wiktionary_url)
    wiktionary_soup = BeautifulSoup(wiktionary_page.content, 'html.parser')
    etymology_section = wiktionary_soup.find('h3', text='Etymology')
    wiktionary_text = etymology_section.find_next('p').get_text(strip=True) if etymology_section else "No data found"

    # Print the result
    # print(f"{word}:\n\nEtymonline: {etymonline_text}\n\nWiktionary: {wiktionary_text}\n")
    prompt_list.append(f"{word}:\n\nEtymonline: {etymonline_text}\n\nWiktionary: {wiktionary_text}\n")
    etymonline_list.append(etymonline_text)
    wiktionary_list.append(wiktionary_text)

print('test')
with open('etymonline_col.txt', 'w') as f:
    f.write('\n'.join(etymonline_list))
with open('wiktionary_col.txt', 'w') as f:
    f.write('\n'.join(wiktionary_list))
with open('prompt_col.txt', 'w') as f:
    f.write('\n'.join(prompt_list))

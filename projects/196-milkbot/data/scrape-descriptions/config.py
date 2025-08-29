# Configuration for the product description scraper

# Number of products to process
MAX_PRODUCTS = 100

# Delay between requests (in seconds)
REQUEST_DELAY = 2

# Page load wait time (in seconds)
PAGE_LOAD_WAIT = 3

# Browser settings
BROWSER_HEADLESS = False  # Set to True for production
BROWSER_TIMEOUT = 30000  # milliseconds

# CSV file paths
INPUT_CSV_PATH = '../scrape-search-results/milk_products.csv'
OUTPUT_CSV_FILENAME = 'milk_products.csv'

# User agent
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# Browser arguments
BROWSER_ARGS = [
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--disable-dev-shm-usage',
    '--disable-web-security',
    '--disable-features=VizDisplayCompositor',
    '--disable-blink-features=AutomationControlled'
]

# HTTP headers
HTTP_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

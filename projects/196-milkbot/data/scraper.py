import asyncio
from pyppeteer import launch
from bs4 import BeautifulSoup
import csv
import time
import re
from urllib.parse import urljoin, urlparse
import json

class WoolworthsScraper:
    def __init__(self):
        self.base_url = "https://www.woolworths.co.nz"
        self.browser = None
        self.page = None
        
    async def init_browser(self):
        """Initialize the browser"""
        print("Launching browser...")
        self.browser = await launch(
            headless=False,  # Set to False for debugging
            args=[
                '--no-sandbox', 
                '--disable-setuid-sandbox', 
                '--disable-dev-shm-usage',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--disable-blink-features=AutomationControlled'
            ]
        )
        print("Browser initialized successfully")
        
        self.page = await self.browser.newPage()
        
        # Set user agent
        await self.page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Set viewport
        await self.page.setViewport({'width': 1920, 'height': 1080})
        
        # Set extra headers
        await self.page.setExtraHTTPHeaders({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
    async def close_browser(self):
        """Close the browser"""
        if self.browser:
            await self.browser.close()
    
    async def get_page_content(self, url):
        """Fetch the page content with JavaScript execution"""
        try:
            print(f"Navigating to: {url}")
            
            # Navigate to the page with shorter timeout
            await self.page.goto(url, {'waitUntil': 'domcontentloaded', 'timeout': 30000})
            print("Page loaded successfully!")
            
            # Wait for the page to fully load
            await asyncio.sleep(3)
            
            # Get the rendered HTML
            content = await self.page.content()
            print(f"Got page content, length: {len(content)}")
            return content
            
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_product_data(self, product_element):
        """Extract data from a single product element"""
        try:
            # Initialize product data
            product_data = {
                'name': '',
                'description': '',
                'current_price': '',
                'was_price': '',
                'savings': '',
                'special_offer': '',
                'unit_price': '',
                'image_url': '',
                'product_url': '',
                'stock_code': ''
            }
            
            # Extract product name from h3 elements (we saw 104 h3 elements in debug)
            title_element = product_element.find('h3')
            if title_element:
                product_data['name'] = title_element.get_text(strip=True)
            
            # Look for price information in the product element
            # Try to find price elements
            price_elements = product_element.find_all(string=re.compile(r'\$\d+\.?\d*'))
            if price_elements:
                for price_text in price_elements:
                    price_match = re.search(r'\$(\d+\.?\d*)', price_text)
                    if price_match:
                        product_data['current_price'] = price_match.group(1)
                        break
            
            # Look for unit price (e.g., $10.67 / 1L)
            unit_price_elements = product_element.find_all(string=re.compile(r'\$\d+\.?\d*\s*/\s*\d+[a-zA-Z]+'))
            if unit_price_elements:
                for unit_text in unit_price_elements:
                    unit_match = re.search(r'\$(\d+\.?\d*)\s*/\s*(\d+[a-zA-Z]+)', unit_text)
                    if unit_match:
                        product_data['unit_price'] = f"${unit_match.group(1)}/{unit_match.group(2)}"
                        break
            
            # Extract image URL
            img_element = product_element.find('img')
            if img_element and img_element.get('src'):
                product_data['image_url'] = img_element['src']
            
            # Extract product URL
            link_element = product_element.find('a', href=True)
            if link_element:
                product_url = link_element['href']
                if product_url.startswith('/'):
                    product_data['product_url'] = urljoin(self.base_url, product_url)
                else:
                    product_data['product_url'] = product_url
                
                # Extract stock code from URL
                stock_code_match = re.search(r'stockcode=(\d+)', product_url)
                if stock_code_match:
                    product_data['stock_code'] = stock_code_match.group(1)
            
            return product_data
            
        except Exception as e:
            print(f"Error extracting product data: {e}")
            return None
    
    async def scrape_products(self, search_url, max_pages=None):
        """Scrape products from the search results"""
        all_products = []
        page = 1
        
        while True:
            if max_pages and page > max_pages:
                break
                
            # Construct page URL
            if 'page=' in search_url:
                page_url = re.sub(r'page=\d+', f'page={page}', search_url)
            else:
                separator = '&' if '?' in search_url else '?'
                page_url = f"{search_url}{separator}page={page}"
            
            print(f"Scraping page {page}: {page_url}")
            
            # Get page content
            content = await self.get_page_content(page_url)
            if not content:
                print(f"Failed to fetch page {page}")
                break
            
            # Parse HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # Based on debug output, look for product containers
            # The products seem to be in some kind of grid or container
            product_elements = []
            
            # Try different approaches to find products
            selectors_to_try = [
                'product-stamp-grid',
                '[data-testid*="product"]',
                '.product',
                '.product-stamp',
                '.product-item',
                'div[class*="product"]',
                'div[class*="stamp"]'
            ]
            
            for selector in selectors_to_try:
                product_elements = soup.select(selector)
                if product_elements:
                    print(f"Found {len(product_elements)} products using selector: {selector}")
                    break
            
            # If no products found with specific selectors, try to find them by looking for h3 elements
            # that contain product names (we saw 104 h3 elements in debug)
            if not product_elements:
                h3_elements = soup.find_all('h3')
                print(f"Found {len(h3_elements)} h3 elements, looking for product containers...")
                
                # Try to find the parent containers of these h3 elements
                for h3 in h3_elements:
                    # Look for a parent container that might be a product
                    parent = h3.parent
                    if parent and parent.name == 'div':
                        # Check if this looks like a product container
                        if parent.find('img') or parent.find('a'):
                            product_elements.append(parent)
                
                print(f"Found {len(product_elements)} potential product containers from h3 parents")
            
            if not product_elements:
                print(f"No products found on page {page}")
                # Save the HTML for debugging
                with open(f'debug_page_{page}.html', 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Saved debug HTML to debug_page_{page}.html")
                break
            
            # Extract data from each product
            for i, product_element in enumerate(product_elements):
                product_data = self.extract_product_data(product_element)
                if product_data and product_data['name']:  # Only add if we got a name
                    all_products.append(product_data)
                    print(f"  Extracted product {i+1}: {product_data['name'][:50]}...")
            
            # Check if there's a next page
            next_page = soup.find('a', string=re.compile(r'Next|next', re.IGNORECASE))
            if not next_page:
                print("No next page found, stopping")
                break
            
            page += 1
            await asyncio.sleep(3)  # Be respectful with requests
        
        return all_products
    
    def save_to_csv(self, products, filename='woolworths_products.csv'):
        """Save products data to CSV file"""
        if not products:
            print("No products to save")
            return
        
        fieldnames = [
            'name', 'description', 'current_price', 'was_price', 'savings',
            'special_offer', 'unit_price', 'image_url', 'product_url', 'stock_code'
        ]
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(products)
            
            print(f"Successfully saved {len(products)} products to {filename}")
            
        except Exception as e:
            print(f"Error saving to CSV: {e}")

async def main():
    # Initialize scraper
    scraper = WoolworthsScraper()
    
    try:
        # Initialize browser
        await scraper.init_browser()
        
        # Search URL
        search_url = "https://www.woolworths.co.nz/shop/searchproducts?search=milk"
        
        print("Starting Woolworths NZ product scraper...")
        print(f"Search URL: {search_url}")
        
        # Scrape products (limit to 2 pages for testing)
        products = await scraper.scrape_products(search_url, max_pages=2)
        
        if products:
            print(f"\nTotal products scraped: {len(products)}")
            
            # Display first few products as preview
            print("\nFirst 3 products preview:")
            for i, product in enumerate(products[:3]):
                print(f"\nProduct {i+1}:")
                for key, value in product.items():
                    print(f"  {key}: {value}")
            
            # Save to CSV
            scraper.save_to_csv(products)
            
        else:
            print("No products were scraped")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Always close the browser
        await scraper.close_browser()

if __name__ == "__main__":
    asyncio.run(main())

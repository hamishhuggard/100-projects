import requests
from bs4 import BeautifulSoup
import csv
import time
import re
from urllib.parse import urljoin, urlparse
import json

class WoolworthsScraper:
    def __init__(self):
        self.base_url = "https://www.woolworths.co.nz"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def get_page_content(self, url):
        """Fetch the page content"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
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
            
            # Extract product name and description
            title_element = product_element.find('h3', {'id': re.compile(r'product-\d+-title')})
            if title_element:
                product_data['name'] = title_element.get_text(strip=True)
            
            # Extract current price
            price_element = product_element.find('h3', {'id': re.compile(r'product-\d+-price')})
            if price_element:
                price_text = price_element.get_text(strip=True)
                # Extract just the price number
                price_match = re.search(r'\$?\s*(\d+\.?\d*)', price_text)
                if price_match:
                    product_data['current_price'] = price_match.group(1)
            
            # Extract was price and savings
            was_price_element = product_element.find('span', class_='price--was')
            if was_price_element:
                was_price_text = was_price_element.get_text(strip=True)
                was_price_match = re.search(r'Was \$(\d+\.?\d*)', was_price_text)
                if was_price_match:
                    product_data['was_price'] = was_price_match.group(1)
            
            save_element = product_element.find('span', class_='price--save')
            if save_element:
                save_text = save_element.get_text(strip=True)
                save_match = re.search(r'Save \$(\d+\.?\d*)', save_text)
                if save_match:
                    product_data['savings'] = save_match.group(1)
            
            # Extract special offer
            special_element = product_element.find('div', class_='productStrap-text')
            if special_element:
                product_data['special_offer'] = special_element.get_text(strip=True)
            
            # Extract unit price
            unit_price_element = product_element.find('span', class_='cupPrice')
            if unit_price_element:
                unit_price_text = unit_price_element.get_text(strip=True)
                unit_price_match = re.search(r'\$(\d+\.?\d*)\s*/\s*(\d+g)', unit_price_text)
                if unit_price_match:
                    product_data['unit_price'] = f"${unit_price_match.group(1)}/{unit_price_match.group(2)}"
            
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
    
    def scrape_products(self, search_url, max_pages=None):
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
            content = self.get_page_content(page_url)
            if not content:
                print(f"Failed to fetch page {page}")
                break
            
            # Parse HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find all product elements
            product_elements = soup.find_all('product-stamp-grid')
            
            if not product_elements:
                print(f"No products found on page {page}")
                break
            
            print(f"Found {len(product_elements)} products on page {page}")
            
            # Extract data from each product
            for product_element in product_elements:
                product_data = self.extract_product_data(product_element)
                if product_data:
                    all_products.append(product_data)
            
            # Check if there's a next page
            next_page = soup.find('a', string=re.compile(r'Next|next', re.IGNORECASE))
            if not next_page:
                print("No next page found, stopping")
                break
            
            page += 1
            time.sleep(1)  # Be respectful with requests
        
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

def main():
    # Initialize scraper
    scraper = WoolworthsScraper()
    
    # Search URL
    search_url = "https://www.woolworths.co.nz/shop/searchproducts?search=milk&page=6&inStockProductsOnly=false"
    
    print("Starting Woolworths NZ product scraper...")
    print(f"Search URL: {search_url}")
    
    # Scrape products (limit to 5 pages for testing)
    products = scraper.scrape_products(search_url, max_pages=5)
    
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

if __name__ == "__main__":
    main()

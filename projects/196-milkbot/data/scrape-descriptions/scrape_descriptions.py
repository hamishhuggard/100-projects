import asyncio
import csv
import os
from pyppeteer import launch
from bs4 import BeautifulSoup
import re
from config import *

class DescriptionScraper:
    def __init__(self):
        self.browser = None
        self.page = None
        
    async def init_browser(self):
        """Initialize the browser"""
        print("Launching browser...")
        self.browser = await launch(
            headless=BROWSER_HEADLESS,
            args=BROWSER_ARGS
        )
        self.page = await self.browser.newPage()
        
        # Set user agent
        await self.page.setUserAgent(USER_AGENT)
        
        # Set viewport
        await self.page.setViewport({'width': 1920, 'height': 1080})
        
        # Set extra headers
        await self.page.setExtraHTTPHeaders(HTTP_HEADERS)
        print("Browser initialized successfully")
        
    async def close_browser(self):
        """Close the browser"""
        if self.browser:
            await self.browser.close()
    
    async def get_page_content(self, url):
        """Fetch the page content with JavaScript execution"""
        try:
            print(f"Navigating to: {url}")
            
            # Navigate to the page with timeout from config
            await self.page.goto(url, {'waitUntil': 'domcontentloaded', 'timeout': BROWSER_TIMEOUT})
            print("Page loaded successfully!")
            
            # Wait for the page to fully load
            await asyncio.sleep(PAGE_LOAD_WAIT)
            
            # Get the rendered HTML
            content = await self.page.content()
            print(f"Got page content, length: {len(content)}")
            return content
            
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_description(self, html_content):
        """Extract product description from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for the product description element with the specific class and attribute
            description_element = soup.find('p', class_='product-description')
            
            if description_element:
                # Get the text content and clean it up
                description_text = description_element.get_text(separator=' ', strip=True)
                
                # Clean up extra whitespace and newlines
                description_text = re.sub(r'\s+', ' ', description_text)
                description_text = description_text.strip()
                
                print(f"Found description: {description_text[:100]}...")
                return description_text
            else:
                # Try alternative selectors if the main one doesn't work
                alternative_selectors = [
                    'p[class*="description"]',
                    'div[class*="description"]',
                    '.product-info p',
                    '.product-details p'
                ]
                
                for selector in alternative_selectors:
                    alt_element = soup.select_one(selector)
                    if alt_element:
                        description_text = alt_element.get_text(separator=' ', strip=True)
                        description_text = re.sub(r'\s+', ' ', description_text)
                        description_text = description_text.strip()
                        
                        if len(description_text) > 20:  # Only use if it's substantial
                            print(f"Found description using alternative selector '{selector}': {description_text[:100]}...")
                            return description_text
                
                print("No product description found")
                return ""
                
        except Exception as e:
            print(f"Error extracting description: {e}")
            return ""

async def main():
    """Main function to scrape product descriptions"""
    print("Starting product description scraper...")
    print(f"Configuration: MAX_PRODUCTS={MAX_PRODUCTS}, REQUEST_DELAY={REQUEST_DELAY}s")
    
    # Initialize scraper
    scraper = DescriptionScraper()
    
    try:
        # Initialize browser
        await scraper.init_browser()
        
        # Read the CSV file
        if not os.path.exists(INPUT_CSV_PATH):
            print(f"CSV file not found: {INPUT_CSV_PATH}")
            print("Current working directory:", os.getcwd())
            print("Available files:", os.listdir('.'))
            return
        
        products_with_descriptions = []
        valid_urls_found = 0
        skipped_products = 0
        
        with open(INPUT_CSV_PATH, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                # Check if we have a valid product URL (not a product group URL)
                if (row['product_url'] and 
                    'productdetails' in row['product_url'] and 
                    valid_urls_found < MAX_PRODUCTS):
                    
                    valid_urls_found += 1
                    print(f"\n--- Processing product {valid_urls_found}/{MAX_PRODUCTS} ---")
                    print(f"Product: {row['name']}")
                    print(f"URL: {row['product_url']}")
                    
                    # Get page content
                    content = await scraper.get_page_content(row['product_url'])
                    
                    if content:
                        # Extract description
                        description = scraper.extract_description(content)
                        
                        # Create product data with description
                        product_data = row.copy()
                        product_data['description'] = description
                        products_with_descriptions.append(product_data)
                        
                        print(f"Successfully extracted description for: {row['name']}")
                    else:
                        print(f"Failed to get content for: {row['name']}")
                        # Add product without description
                        products_with_descriptions.append(row)
                    
                    # Wait between requests (except for the last one)
                    if valid_urls_found < MAX_PRODUCTS:
                        print(f"Waiting {REQUEST_DELAY} seconds before next product...")
                        await asyncio.sleep(REQUEST_DELAY)
                    
                    if valid_urls_found >= MAX_PRODUCTS:
                        break
                else:
                    if row['product_url'] and 'productgroup' in row['product_url']:
                        skipped_products += 1
                        print(f"Skipping product group URL: {row['name']}")
                    elif not row['product_url']:
                        skipped_products += 1
                        print(f"Skipping product without URL: {row['name']}")
        
        # Save results with descriptions
        if products_with_descriptions:
            print(f"\nTotal products processed: {len(products_with_descriptions)}")
            print(f"Products skipped: {skipped_products}")
            
            # Save to new CSV with descriptions
            fieldnames = [
                'name', 'description', 'current_price', 'was_price', 'savings',
                'special_offer', 'unit_price', 'image_url', 'product_url', 'stock_code'
            ]
            
            with open(OUTPUT_CSV_FILENAME, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(products_with_descriptions)
            
            print(f"Successfully saved {len(products_with_descriptions)} products with descriptions to {OUTPUT_CSV_FILENAME}")
            
            # Display sample results
            print("\nSample results:")
            for i, product in enumerate(products_with_descriptions[:3]):
                print(f"\nProduct {i+1}: {product['name']}")
                if product['description']:
                    print(f"Description: {product['description'][:150]}...")
                else:
                    print("Description: [No description found]")
        else:
            print("No products were processed")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Always close the browser
        await scraper.close_browser()

if __name__ == "__main__":
    asyncio.run(main())

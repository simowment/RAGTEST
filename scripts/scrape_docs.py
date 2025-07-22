#!/usr/bin/env python3
"""
Advanced script to scrape all sections and sub-packages of Vectorbt's API documentation.
"""

import os
import sys
import requests
import time
import argparse
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Constants
OUTPUT_DIR = Path("../docs_vbt_clean")
BASE_URL = "https://vectorbt.dev/api/"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

def get_page_content(url):
    """Get HTML content from a URL."""
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}", file=sys.stderr)
        return None

def extract_api_links(html_content, base_url):
    """Extract all unique API documentation links from the navigation."""
    if not html_content:
        return []
    
    soup = BeautifulSoup(html_content, 'html.parser')
    links = set()

    # Find the main navigation container for the API section
    nav = soup.find('nav', {'class': 'md-nav--primary'})
    if not nav:
        print("Could not find primary navigation menu.", file=sys.stderr)
        return []

    # Find all links within the API navigation section
    for a_tag in nav.find_all('a', href=True):
        href = a_tag['href']
        # Construct the full URL
        full_url = urljoin(base_url, href)
        
        # We only want links that are part of the API documentation
        if "/api/" in full_url:
            links.add(full_url)
            
    return sorted(list(links))

def extract_main_content(html_content):
    """Extract the main textual content from a documentation page."""
    if not html_content:
        return "", ""
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract title
    title_tag = soup.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    # Extract main content
    content_area = soup.find('article', {'class': 'md-content__inner'})
    if not content_area:
        return title, ""

    # Remove non-content elements like scripts, styles, etc.
    for element in content_area(["script", "style", "nav", "footer", ".md-source"]):
        element.decompose()

    text = content_area.get_text(separator='\n', strip=True)
    return title, text

def save_to_file(title, content, output_dir):
    """Save content to a file with a sanitized filename."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanitize the title to create a valid filename
    filename = "".join(c for c in title if c.isalnum() or c in (' ', '.', '-')).rstrip()
    filename = filename.replace(' ', '_').replace('.', '_').lower() + ".txt"
    
    output_path = output_dir / filename
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return output_path
    except IOError as e:
        print(f"Error writing to file {output_path}: {e}", file=sys.stderr)
        return None

def scrape_site(start_url, output_dir, delay=1.0):
    """Scrape the entire Vectorbt API documentation."""
    print(f"Starting scrape of {start_url}")

    # First, get the main page to find all the links
    main_page_html = get_page_content(start_url)
    if not main_page_html:
        print("Failed to retrieve the main API page. Aborting.", file=sys.stderr)
        return

    links_to_scrape = extract_api_links(main_page_html, start_url)
    if not links_to_scrape:
        print("No API links found to scrape. Check the selectors.", file=sys.stderr)
        return
        
    print(f"Found {len(links_to_scrape)} pages to scrape.")
    
    scraped_count = 0
    for url in links_to_scrape:
        print(f"Scraping: {url}")
        
        page_html = get_page_content(url)
        if not page_html:
            continue
            
        title, content = extract_main_content(page_html)
        
        if content:
            output_path = save_to_file(title, content, output_dir)
            if output_path:
                print(f"  -> Saved to {output_path}")
                scraped_count += 1
        else:
            print(f"  -> No content found for {url}")
            
        time.sleep(delay)

    print(f"\nScraping complete. Successfully scraped {scraped_count} pages.")

def main():
    parser = argparse.ArgumentParser(description="Scrape the Vectorbt API documentation.")
    parser.add_argument("--output", "-o", default=str(OUTPUT_DIR),
                        help="Output directory for the scraped text files.")
    parser.add_argument("--start-url", "-u", default=BASE_URL,
                        help="The base URL to start scraping from.")
    parser.add_argument("--delay", "-d", type=float, default=0.5,
                        help="Delay in seconds between fetching pages.")
    
    args = parser.parse_args()
    
    scrape_site(args.start_url, Path(args.output), args.delay)

if __name__ == "__main__":
    main() 
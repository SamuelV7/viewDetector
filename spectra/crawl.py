import requests
from bs4 import BeautifulSoup
import time
import random
import requests
# from readability import Document
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
from datetime import datetime

def fetch_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to retrieve content from {url}. Status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def get_beautiful_soup_object(html):
    return BeautifulSoup(html, 'html.parser')

def extract_valuable_text_and_links(soup: BeautifulSoup, base_url):
    # Remove unnecessary tags like script, style, footer, header, and navigation
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        tag.decompose()

    # Extract valuable text by getting all paragraphs and other relevant text content
    valuable_text = ' '.join([element.get_text(strip=True) for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])])
    
    # Extract all hyperlinks on the page
    links = []
    for a_tag in soup.find_all('a', href=True):
        link = urljoin(base_url, a_tag['href'])  # Convert relative links to absolute links
        links.append(link)

    return valuable_text, links

@dataclass
class Page:
    content: str
    article: bool
    links: list

def scrape_page(soup: BeautifulSoup, article:bool) -> Page:
    html_content = fetch_html(url)
    # page = None
    if html_content:
        valuable_text, links = extract_valuable_text_and_links(soup, url)
        page = Page(content=valuable_text, article=article, links=links)
        if article:
            print("Extracted Valuable Text:\n")
            print(valuable_text[:250] + "...")  # Print first 500 characters of valuable text (for brevity)
            print(f"\nExtracted Links: {len(links)} \n")
        # for link in links:
        #     print(link)
        return page

def is_article_based_on_length(soup) -> bool:
    paragraphs = soup.find_all('p')
    text_length = sum(len(p.get_text()) for p in paragraphs)
    # Assuming an article has at least 1000 characters of text
    return text_length > 1000

def save_pages(name:str, pages, output_file):
    dict_pages = {name: [], "non_article": []}
    for i, page in enumerate(pages):
        if page.article:
            dict_pages[name].append({"content": page.content, "links": page.links})
        else:
            dict_pages["non_article"].append({"content": page.content, "links": page.links})
    with open(output_file, 'w') as file:
        json.dump(dict_pages, file, indent=4)

def check_if_url_is_valid(url: str) -> bool:
    return url.startswith("http") or url.startswith("www") or url.startswith("https") or url.endswith(".com") or url.endswith(".org")

def crawl(url, save_name):
    visited_links = {}
    to_visit = [url]
    # depth_limit = 3
    # depth = 0
    articles_total = 0
    articles = []
    while (to_visit.__len__ != 0) and articles_total < 100:
        current_url = to_visit.pop(0)
        if current_url in visited_links:
            continue
        if not check_if_url_is_valid(current_url):
            continue
        visited_links[current_url] = True
        soup = get_beautiful_soup_object(fetch_html(current_url))
        is_article = is_article_based_on_length(soup)
        if is_article:
            articles_total += 1
        textLink : Page = scrape_page(soup, is_article)
        articles.append(textLink)
        # add the page to the list of pages to visit if it is not already visited
        to_visit += [link for link in textLink.links if link not in visited_links]
        # depth += 1
    save_pages(save_name, articles, f"{save_name}@{datetime.now()}.json")

if __name__ == "__main__":
    # url = "https://edition.cnn.com" #Replace with your target URL
    url = "https://www.foxnews.com"
    crawl(url, "foxnews")
            

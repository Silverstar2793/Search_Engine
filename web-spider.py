import requests
from bs4 import BeautifulSoup
import sqlite3


def crawler(start_url, max_pages=100):

    # Connect to the SQLite database
    conn = sqlite3.connect("crawled_pages.db")
    c = conn.cursor()

    # Create the `pages` table with the corrected schema
    c.execute(
        """CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            content TEXT,
            cleaned_content TEXT,
            title TEXT,
            outgoing_links TEXT,
            pagerank REAL
        )"""
    )
    conn.commit()

    # Initialize the frontier and visited sets
    url_frontier = [start_url]
    visited_pages = set()

    while url_frontier and len(visited_pages) < max_pages:
        url = url_frontier.pop(0)
        if url in visited_pages:
            continue

        print(f"Crawling: {url}")
        try:
            # Fetch the page content
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"Failed to fetch {url} with status code {response.status_code}")
                continue

            soup = BeautifulSoup(response.content, "html.parser")
            title = soup.find("title").string if soup.find("title") else "No Title"

            # Collect outgoing links
            outgoing_links = []
            for link in soup.find_all("a"):
                href = link.get("href")
                if href and href.startswith("http"):
                    outgoing_links.append(href)

            # Insert data into the database
            c.execute(
                """INSERT OR REPLACE INTO pages 
                (url, content, cleaned_content, title, outgoing_links) 
                VALUES (?, ?, ?, ?, ?)""",
                (url, str(soup), soup.get_text(), title, ','.join(outgoing_links)),
            )
            conn.commit()

            # Add new links to the frontier
            for href in outgoing_links:
                if href not in visited_pages and href not in url_frontier:
                    url_frontier.append(href)

            visited_pages.add(url)
        except Exception as e:
            print(f"Error while crawling {url}: {e}")

    # Close the database connection
    conn.close()
    print("Crawling completed")


# Seed URLs to start the crawling process
seed_urls = ["https://www.bbc.co.uk/news/topics/cp29jzed52et", "https://www.cnn.com"]
for url in seed_urls:
    crawler(url, 50)

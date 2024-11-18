import sqlite3
import networkx as nx

# Connect to the SQLite database
conn = sqlite3.connect("crawled_pages.db")
cursor = conn.cursor()

# Retrieve the URLs of all the websites from the database
cursor.execute("SELECT url FROM pages")
urls = [row[0] for row in cursor.fetchall()]

# Create a directed graph empty
graph = nx.DiGraph()
# urls = [row[0] for row in cursor.fetchall()]

for url in urls:
    graph.add_node(url)

for url in urls:
    cursor.execute("SELECT outgoing_links FROM pages WHERE url = ?", (url,))
    outgoing_links = cursor.fetchone()[0].split(",")
    for link in outgoing_links:
        if link.startswith("http"):
            graph.add_edge(url, link)

# Calculate the PageRank of each website
pagerank = nx.pagerank(graph)

# Store the PageRank of each website in the database
for url in urls:
    cursor.execute("UPDATE pages SET pagerank = ? WHERE url = ?", (pagerank[url], url))
    conn.commit()
conn.close

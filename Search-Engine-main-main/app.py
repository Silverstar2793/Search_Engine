import spacy
from nltk.corpus import stopwords
from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import re
import string
from collections import defaultdict
from math import log
from colorama import Fore, Style
import numpy as np
from rapidfuzz import fuzz, process
import time

# Initialize the Flask app
app = Flask(__name__)

# Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Initialize stop words
stop_words = set(stopwords.words('english'))

def fetch_website_content(url):
    """
    Fetch and extract the main text content from a website.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs if para.get_text()])
        return content
    except requests.RequestException as e:
        print(f"Failed to fetch content from {url}: {e}")
        return None

def normalize_terms(text):
    """
    Normalize terms by handling common variations and abbreviations.
    """
    term_patterns = [
        (r'\bai\b|\bartificial intelligence\b', 'artificial intelligence'),
        (r'\bml\b|\bmachine learning\b', 'machine learning'),
        (r'\bdl\b|\bdeep learning\b', 'deep learning'),
        (r'\bnlp\b|\bnatural language processing\b', 'natural language processing'),
        (r'\biot\b|\binternet of things\b', 'internet of things')
    ]
    
    text = text.lower()
    for pattern, replacement in term_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def preprocess(text):
    """
    Preprocess text: normalize terms, tokenize, remove stopwords, and lemmatize.
    """
    normalized_text = normalize_terms(text)
    doc = nlp(normalized_text)
    processed_tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.text.strip()]
    return processed_tokens

# List of websites to scrape
websites = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Climate_change",
    "https://en.wikipedia.org/wiki/Quantum_computing",
    "https://en.wikipedia.org/wiki/Sustainability",
    "https://en.wikipedia.org/wiki/Healthcare"
]

# Step 1: Scrape and Preprocess Web Content
web_documents = {}
for i, url in enumerate(websites):
    content = fetch_website_content(url)
    if content:
        web_documents[url] = preprocess(content)

# Step 2: Build an Inverted Index
inverted_index = defaultdict(list)
for url, tokens in web_documents.items():
    for token in tokens:
        if url not in inverted_index[token]:
            inverted_index[token].append(url)

# Step 3: Compute TF-IDF Scores
def compute_tf_idf(docs, index):
    N = len(docs)
    tf_idf = {url: {} for url in docs}
    for term, urls in index.items():
        idf = log(N / len(urls))
        for url in urls:
            tf = docs[url].count(term) / len(docs[url])
            tf_idf[url][term] = tf * idf
    return tf_idf

tf_idf = compute_tf_idf(web_documents, inverted_index)

# Step 4: Vectorize Documents and Query
def vectorize(tf_idf, query_terms):
    terms = list(set(query_terms + [term for doc in tf_idf.values() for term in doc]))
    vectors = {url: [doc.get(term, 0) for term in terms] for url, doc in tf_idf.items()}
    query_vector = [1 if term in query_terms else 0 for term in terms]
    return vectors, query_vector

# Step 5: Calculate Cosine Similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query_text = ""
    search_time = None

    if request.method == "POST":
        start_time = time.time()
        query_text = request.form.get("query")
        query_tokens = preprocess(query_text)
        
        # Vectorize the query and compute similarity scores
        vectors, query_vector = vectorize(tf_idf, query_tokens)
        scores = {url: cosine_similarity(query_vector, vector) for url, vector in vectors.items()}
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        # Prepare the results with a snippet
        for url, score in sorted_scores:
            if score > 0:
                # Fetch a snippet from the original content (first 150 characters)
                original_content = ' '.join(web_documents[url][:30]) if url in web_documents else "No content available"
                snippet = original_content[:150] + "..." if len(original_content) > 150 else original_content
                
                # Append the URL (doc_id), snippet (text), and score
                results.append((url, snippet, score))
        
        end_time = time.time()
        search_time = round((end_time - start_time) * 1000, 2)

    return render_template("index.html", results=results, query_text=query_text, search_time=search_time)

if __name__ == "__main__":
    app.run(debug=True)

import spacy
from flask import Flask, render_template, request
import re
import numpy as np
from collections import defaultdict
from math import log
from nltk.corpus import wordnet as wn
import time

# Initialize Flask app
app = Flask(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Example documents
documents = {
    "doc1": "Climate change is one of the most pressing issues of our time. Global warming impacts weather patterns.",
    "doc2": "Artificial intelligence is rapidly changing industries, with applications in healthcare, finance, and beyond.",
    "doc3": "Quantum computing promises to revolutionize the field of computing by solving complex problems much faster.",
    "doc4": "Sustainability efforts are vital to combating climate change. Renewable energy sources are part of the solution.",
    "doc5": "Artificial intelligence can assist in predicting medical conditions based on patient data and improve healthcare outcomes."
}

# Preprocessing Function
def preprocess(text):
    text = text.lower()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return tokens

# Indexing
inverted_index = defaultdict(list)
document_tokens = {}
for doc_id, content in documents.items():
    tokens = preprocess(content)
    document_tokens[doc_id] = tokens
    for token in set(tokens):
        inverted_index[token].append(doc_id)

# TF-IDF Calculation
def compute_tf_idf(tokens, doc_id):
    term_freq = {token: tokens.count(token) / len(tokens) for token in tokens}
    doc_count = len(documents)
    tf_idf = {}
    for token, tf in term_freq.items():
        idf = log(doc_count / len(inverted_index[token]))
        tf_idf[token] = tf * idf
    return tf_idf

document_vectors = {doc_id: compute_tf_idf(tokens, doc_id) for doc_id, tokens in document_tokens.items()}

# Vectorize Query and Documents
def vectorize_query(query_tokens):
    terms = list(set(query_tokens))
    query_vector = [1 if term in query_tokens else 0 for term in terms]
    doc_vectors = {doc_id: [doc.get(term, 0) for term in terms] for doc_id, doc in document_vectors.items()}
    return query_vector, doc_vectors

# Cosine Similarity
def cosine_similarity(query_vector, doc_vector):
    dot_product = np.dot(query_vector, doc_vector)
    norm_query = np.linalg.norm(query_vector)
    norm_doc = np.linalg.norm(doc_vector)
    return dot_product / (norm_query * norm_doc) if norm_query and norm_doc else 0

# Synonym Expansion
def expand_query(query):
    query_tokens = preprocess(query)
    expanded_query = set(query_tokens)
    for token in query_tokens:
        for syn in wn.synsets(token):
            for lemma in syn.lemmas():
                expanded_query.add(lemma.name())
    return list(expanded_query)

# Search Route
@app.route("/", methods=["GET", "POST"])
def search():
    query = ""
    results = []
    search_time = None

    if request.method == "POST":
        query = request.form.get("query")
        start_time = time.time()

        # Expand query with synonyms
        expanded_query = expand_query(query)

        # Vectorize query and documents
        query_vector, doc_vectors = vectorize_query(expanded_query)

        # Compute similarity scores
        scores = {
            doc_id: cosine_similarity(query_vector, doc_vector)
            for doc_id, doc_vector in doc_vectors.items()
        }

        # Sort results by score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Prepare results with highlights
        results = [
            (doc_id, documents[doc_id], score)
            for doc_id, score in sorted_results if score > 0
        ]

        search_time = round((time.time() - start_time) * 1000, 2)

    return render_template("index.html", query=query, results=results, search_time=search_time)

# Run the App
if __name__ == "__main__":
    app.run(debug=True)
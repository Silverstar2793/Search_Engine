import spacy
from nltk.corpus import stopwords
from flask import Flask, render_template, request
import re
import string
from collections import defaultdict
from math import log
from tabulate import tabulate
from colorama import Fore, Style
import numpy as np
from rapidfuzz import fuzz, process
import time

# Initialize the Flask app
app = Flask(__name__)

# Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Example documents
documents = {
    "doc1": "Climate change is one of the most pressing issues of our time. Global warming impacts weather patterns.",
    "doc2": "Artificial intelligence is rapidly changing industries, with applications in healthcare, finance, and beyond.",
    "doc3": "Quantum computing promises to revolutionize the field of computing by solving complex problems much faster.",
    "doc4": "Sustainability efforts are vital to combating climate change. Renewable energy sources are part of the solution.",
    "doc5": "Artificial intelligence can assist in predicting medical conditions based on patient data and improve healthcare outcomes."
}

# Initialize stop words
stop_words = set(stopwords.words('english'))

class SpellChecker:
    def __init__(self, documents):
        """Initialize spell checker with vocabulary from documents"""
        # Extract all words from documents to build vocabulary
        self.vocabulary = set()
        for doc in documents.values():
            # Convert to lowercase and split into words
            words = re.findall(r'\b\w+\b', doc.lower())
            self.vocabulary.update(words)
        
        print(f"Vocabulary size: {len(self.vocabulary)} words")

    def correct_spelling(self, word, similarity_threshold=80):
        """
        Correct spelling using fuzzy matching
        Returns the most similar word if similarity is above threshold
        """
        if word.lower() in self.vocabulary:
            return word
        
        # Use fuzzy matching to find the closest match in vocabulary
        result = process.extractOne(
            word,
            self.vocabulary,
            scorer=fuzz.ratio,
            score_cutoff=similarity_threshold
        )
        
        if result is not None:
            # Handle the tuple returned by extractOne
            suggested_word = result[0]  # First element is the matched word
            similarity = result[1]      # Second element is the similarity score
            print(f"Corrected '{word}' to '{suggested_word}' (similarity: {similarity}%)")
            return suggested_word
        
        return word

def normalize_terms(text):
    """
    Normalize terms by handling common variations and abbreviations dynamically.
    """
    # Common patterns for terms and their variations
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

def extract_abbreviations(text):
    """
    Dynamically extract potential abbreviations and their expansions from text.
    """
    abbr_pattern = r'\b([A-Za-z]+)\s*\(([A-Z]+)\)\b|\b([A-Z]+)\s*\(([A-Za-z\s]+)\)\b'
    matches = re.finditer(abbr_pattern, text)
    
    extracted_terms = {}
    for match in matches:
        if match.group(1) and match.group(2):
            term = match.group(1).lower()
            abbr = match.group(2).lower()
        elif match.group(3) and match.group(4):
            term = match.group(4).lower()
            abbr = match.group(3).lower()
        else:
            continue
        extracted_terms[abbr] = term
    
    return extracted_terms

# Initialize spell checker with documents
spell_checker = SpellChecker(documents)

def preprocess(text, correct_spelling=True):
    print(f"\n{Fore.BLUE}Original Text:{Style.RESET_ALL} {Fore.CYAN}{text}{Style.RESET_ALL}")
    
    # Extract abbreviations
    extracted_terms = extract_abbreviations(text)
    
    # Normalize terms
    normalized_text = normalize_terms(text)
    print(f"{Fore.BLUE}After Term Normalization:{Style.RESET_ALL} {Fore.CYAN}{normalized_text}{Style.RESET_ALL}")
    
    # Remove parenthetical expressions
    normalized_text = re.sub(r'\([^)]*\)', '', normalized_text)
    
    # Tokenize and correct spelling if enabled
    tokens = re.findall(r'\b\w+\b', normalized_text.lower())
    if correct_spelling:
        corrected_tokens = []
        for token in tokens:
            corrected = spell_checker.correct_spelling(token)
            if corrected != token:
                print(f"{Fore.YELLOW}Corrected spelling: {token} -> {corrected}{Style.RESET_ALL}")
            corrected_tokens.append(corrected)
        tokens = corrected_tokens
    
    # Join tokens back into text for lemmatization with spaCy
    normalized_text = ' '.join(tokens)
    
    # Lemmatize tokens and remove stopwords
    doc = nlp(normalized_text)
    processed_tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.text.strip()]
    
    # Display the processed tokens
    formatted_tokens = ', '.join([f"{Fore.YELLOW}{token}{Style.RESET_ALL}" for token in processed_tokens])
    print(f"{Fore.GREEN}After Processing:{Style.RESET_ALL} [{formatted_tokens}]")
    
    return processed_tokens

# Step 1: Preprocess all documents to prepare for indexing
preprocessed_docs = {doc_id: preprocess(text, correct_spelling=False) for doc_id, text in documents.items()}

# Step 2: Build an Inverted Index for quick term-to-document lookups
inverted_index = defaultdict(list)
for doc_id, tokens in preprocessed_docs.items():
    for token in tokens:
        if doc_id not in inverted_index[token]:
            inverted_index[token].append(doc_id)

def display_inverted_index(index):
    print("\nInverted Index (Original Order):\n")
    headers = ["Term", "Document IDs"]
    
    unsorted_table = [[f"{Fore.CYAN}{term}{Style.RESET_ALL}", f"{Fore.GREEN}{', '.join(index[term])}{Style.RESET_ALL}"]
                      for term in index]
    print(tabulate(unsorted_table, headers=headers, tablefmt="fancy_grid"))
    
    # Sort and display inverted index by term
    sorted_index = dict(sorted(index.items()))
    print("\nInverted Index (Sorted by Term):\n")
    sorted_table = [[f"{Fore.CYAN}{term}{Style.RESET_ALL}", f"{Fore.GREEN}{', '.join(sorted_index[term])}{Style.RESET_ALL}"]
                    for term in sorted_index]
    print(tabulate(sorted_table, headers=headers, tablefmt="fancy_grid"))
display_inverted_index(inverted_index)

def compute_tf_idf(docs, index):
    N = len(docs)
    tf_idf = {doc_id: {} for doc_id in docs}
    for term, doc_ids in index.items():
        idf = log(N / len(doc_ids))
        for doc_id in doc_ids:
            tf = docs[doc_id].count(term) / len(docs[doc_id])
            tf_idf[doc_id][term] = tf * idf
    return tf_idf

# Display TF-IDF scores in a tabular format
def display_tf_idf(tf_idf):
    headers = [f"{Fore.BLUE}Document{Style.RESET_ALL}", 
               f"{Fore.MAGENTA}Term{Style.RESET_ALL}", 
               f"{Fore.GREEN}TF-IDF Score{Style.RESET_ALL}"]
    table = []

    for doc_id, term_scores in tf_idf.items():
        for term, score in term_scores.items():
            colored_doc_id = f"{Fore.CYAN}{doc_id}{Style.RESET_ALL}"
            colored_term = f"{Fore.YELLOW}{term}{Style.RESET_ALL}"
            colored_score = f"{Fore.GREEN}{round(score, 4)}{Style.RESET_ALL}"
            table.append([colored_doc_id, colored_term, colored_score])

    print("\nTF-IDF Scores Table:")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

# Calculate and display TF-IDF
tf_idf = compute_tf_idf(preprocessed_docs, inverted_index)
display_tf_idf(tf_idf)

# Create document and query vectors for cosine similarity calculation.
def vectorize(tf_idf, query_terms):
    terms = list(set(query_terms + [term for doc in tf_idf.values() for term in doc]))
    vectors = {doc_id: [doc.get(term, 0) for term in terms] for doc_id, doc in tf_idf.items()}
    query_vector = [1 if term in query_terms else 0 for term in terms]
    return vectors, query_vector

# Calculate cosine similarity between document vectors and query vector.
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

def highlight_terms(text, search_terms):
    """
    Highlight search terms in the text while preserving case sensitivity
    """
    if not search_terms:
        return text
    
    # Create a pattern that matches any of the search terms (case insensitive)
    pattern = r'\b(' + '|'.join(map(re.escape, search_terms)) + r')\b'
    
    def replace_match(match):
        """Replace matched term with highlighted version while preserving original case"""
        return f'<span class="highlight">{match.group(0)}</span>'
    
    # Perform the replacement
    highlighted_text = re.sub(pattern, replace_match, text, flags=re.IGNORECASE)
    return highlighted_text

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query_text = ""
    corrections_made = []
    search_time = None

    if request.method == "POST":
        # Start timing
        start_time = time.time()
        query_text = request.form.get("query")
        
        # Track original query terms for comparison
        original_terms = set(re.findall(r'\b\w+\b', query_text.lower()))
        
        # Process query with spell checking
        query_tokens = preprocess(query_text, correct_spelling=True)
        
        # Track which terms were corrected
        processed_terms = set(query_tokens)
        for orig_term in original_terms:
            corrected_term = spell_checker.correct_spelling(orig_term)
            if corrected_term != orig_term:
                corrections_made.append(f"{orig_term} → {corrected_term}")
        
        print("\nProcessed Query Tokens:", query_tokens)
        
        vectors, query_vector = vectorize(tf_idf, query_tokens)
        scores = {doc_id: cosine_similarity(query_vector, vector) for doc_id, vector in vectors.items()}
        
        print("\nCosine Similarity Scores:")
        for doc_id, score in scores.items():
            print(f"{doc_id}: {score}")
        
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        # Highlight matching terms in results
        results = []
        for doc_id, score in sorted_scores:
            if score > 0:
                highlighted_text = highlight_terms(documents[doc_id], query_tokens)
                results.append((doc_id, highlighted_text, score))
        
        # Calculate search time
        end_time = time.time()
        search_time = round((end_time - start_time) * 1000, 2)  # Convert to milliseconds
        print(f"\nSearch completed in {search_time} ms")

    return render_template(
        "index.html",
        results=results,
        query_text=query_text,
        corrections_made=corrections_made,
        search_time=search_time

    )

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

app = Flask(__name__, template_folder="./static/")


@app.route("/")
def websearch():
    return render_template("websearch.html")


@app.route("/a")
def a():
    return render_template("A.html")


@app.route("/b")
def b():
    return render_template("B.html")


@app.route("/c")
def c():
    return render_template("C.html")


@app.route("/d")
def d():
    return render_template("D.html")


@app.route("/e")
def e():
    return render_template("E.html")


@app.route("/websearch", methods=["GET", "POST"])
def web_search():
    if request.method == "POST":
        query = request.form["query"]
        if query == "":
            return render_template("websearch.html", query=query)
        websites = [
            "http://localhost:5000/a",
            "http://localhost:5000/b",
            "http://localhost:5000/c",
            "http://localhost:5000/d",
            "http://localhost:5000/e",
        ]
        tokenized_text = load_tokenized_text("tokenized_text_pickle.pkl")
       
        # TF-IDF Vectorization
        tfidf = TfidfVectorizer()
        tfidf_vectors = tfidf.fit_transform([' '.join(token) for token in tokenized_text])
        
        # Cosine Similarity
        query_vector = tfidf.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_vectors)
        # print("Cosine Similarity Scores:",similarities)

        if all_zeros(similarities[0]):
            return render_template("notfound.html")

        # Create a graph
        G = nx.DiGraph()  # Use DiGraph for directed graph, Graph for undirected graph. G = nx.G    graph = nx.DiGraph()
        for i, link in enumerate(websites):
            G.add_node(link)
            for j, sim in enumerate(similarities[0]):
                if sim > 0:
                    G.add_edge(link, websites[j], weight=sim)

        # Calculate PageRank
        pagerank = nx.pagerank(G)
        ranked_results = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        # print("Ranked Results:", ranked_results)

        top_results = [x[0] for x in ranked_results if x[1] >= 0.14]

        return render_template("results.html", data=[top_results, query])

  
def load_tokenized_text(filename):
    tokenized_text = pickle.load(open(filename, "rb"))
    return tokenized_text

def all_zeros(l):
    for i in l:
        if i != 0:
            return False
    return True

if __name__ == "__main__":
    app.run(debug=True)

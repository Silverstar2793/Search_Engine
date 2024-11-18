from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__, template_folder='./static')

@app.route("/")
def home():
    return render_template("websearch.html")

@app.route("/websearch", methods=["GET", "POST"])
def search():
    query = request.form['query']

    if query == "":
        return render_template("websearch.html")

    # Connect to the SQLite database
    conn = sqlite3.connect("crawled_pages.db")
    cursor = conn.cursor()

    # Search the database for the query
    cursor.execute("SELECT url, title, pagerank FROM pages WHERE cleaned_content \
                    LIKE ? ORDER BY pagerank DESC", ('%' + query + '%',))
    urls = cursor.fetchall()

    conn.close()

    # Pass PageRank along with the results
    return render_template('results.html', urls=urls, query=query)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request

app = Flask(__name__, template_folder='./static/')


@app.route('/')
def websearch():
    return render_template("websearch.html")

@app.route('/a')
def a():
    return render_template("A.html")

@app.route('/b')
def b():
    return render_template("B.html")

@app.route('/c')
def c():
    return render_template("C.html")

@app.route('/d')
def d():
    return render_template("D.html")

@app.route('/e')
def e():
    return render_template("E.html")



@app.route('/websearch', methods=['GET', 'POST'])
def web_search():
    if request.method == 'POST':
        query = request.form['query']
        if query =="":
            return render_template("websearch.html", query=query)
        return render_template("results.html", data = query)


if __name__ == '__main__':
    app.run(debug=True)
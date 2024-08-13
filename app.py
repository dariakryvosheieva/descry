from PIL import Image
from flask import Flask, render_template, request

from descry import Reader


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html", pred="(will appear here)")


@app.route("/", methods=["post"])
def select_script():
    script = request.form["script"]
    image = Image.open(request.files["file"])
    return render_template("index.html", pred=Reader(script).readtext(image))

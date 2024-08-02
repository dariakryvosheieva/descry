from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

adlam_cnn = tf.keras.models.load_model('models/adlam.h5')
nko_cnn = tf.keras.models.load_model('models/nko.h5')
kayahli_cnn = tf.keras.models.load_model('models/kayahli.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/adlam')
def adlam():
    return render_template('adlam.html')

@app.route('/nko')
def nko():
    return render_template('nko.html')

@app.route('/kayahli')
def kayahli():
    return render_template('kayahli.html')

@app.route('/adlam', methods=['post'])
def process_adlam():
    img = request.files['file']
    img = Image.open(img)
    img = img.convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.reshape((1, 28, 28, 1))
    img = img / 255.

    prediction = adlam_cnn.predict(img)
    c = np.argmax(prediction, axis = 1)

    uppercase = ['𞤀', '𞤁', '𞤂', '𞤃', '𞤄', '𞤅', '𞤆', '𞤇', '𞤈', '𞤉', '𞤊', '𞤋', '𞤌', '𞤍', '𞤎', '𞤏', '𞤐', '𞤑', '𞤒', '𞤓', '𞤔', '𞤕', '𞤖', '𞤗', '𞤘', '𞤙', '𞤚', '𞤛']
    lowercase = ['𞤢', '𞤣', '𞤤', '𞤥', '𞤦', '𞤧', '𞤨', '𞤩', '𞤪', '𞤫', '𞤬', '𞤭', '𞤮', '𞤯', '𞤰', '𞤱', '𞤲', '𞤳', '𞤴', '𞤵', '𞤶', '𞤷', '𞤸', '𞤹', '𞤺', '𞤻', '𞤼', '𞤽']
    ipa = ['a', 'd', 'l', 'm', 'b', 's', 'p', 'ɓ', 'r', 'e', 'f', 'i', 'ɔ', 'ɗ', 'ʔʲ', 'w', 'n', 'k', 'j', 'u', 'd͡ʒ', 't͡ʃ', 'h', 'q', 'g', 'ɲ', 't', 'ŋ']

    return render_template('adlam.html', pred=f'{uppercase[c[0]]}/{lowercase[c[0]]} /{ipa[c[0]]}/')

@app.route('/nko', methods=['post'])
def process_nko():
    img = request.files['file']
    img = Image.open(img)
    img = img.convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.reshape((1, 28, 28, 1))
    img = img / 255.

    prediction = nko_cnn.predict(img)
    c = np.argmax(prediction, axis = 1)

    characters = ['ߊ', 'ߋ', 'ߌ', 'ߍ', 'ߎ', 'ߏ', 'ߐ', 'ߓ', 'ߔ', 'ߖ', 'ߗ', 'ߘ', 'ߕ', 'ߙ', 'ߚ', 'ߛ', 'ߝ', 'ߞ', 'ߟ', 'ߜ', 'ߡ', 'ߢ', 'ߣ', 'ߥ', 'ߦ', 'ߤ', 'ߒ']
    ipa = ['a', 'e', 'i', 'ɛ', 'u', 'o', 'ɔ', 'b', 'p', 'd͡ʒ', 't͡ʃ', 'd', 't', 'r', 'rr', 's', 'f', 'k', 'l', 'g͡b', 'm', 'ɲ', 'n', 'w', 'j', 'h', 'ŋ']

    return render_template('nko.html', pred=f'{characters[c[0]]} /{ipa[c[0]]}/')

@app.route('/kayahli', methods=['post'])
def process_kayahli():
    img = request.files['file']
    img = Image.open(img)
    img = img.convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.reshape((1, 28, 28, 1))
    img = img / 255.

    prediction = kayahli_cnn.predict(img)
    c = np.argmax(prediction, axis = 1)

    characters = ['ꤊ', 'ꤋ', 'ꤌ', 'ꤍ', 'ꤎ', 'ꤏ', 'ꤐ', 'ꤑ', 'ꤒ', 'ꤓ', 'ꤔ', 'ꤕ', 'ꤖ', 'ꤗ', 'ꤘ', 'ꤙ', 'ꤚ', 'ꤛ', 'ꤜ', 'ꤝ', 'ꤞ', 'ꤟ', 'ꤠ', 'ꤡ', 'ꤢ', 'ꤣ', 'ꤤ', 'ꤥ', 'ꤢꤦ', 'ꤢꤧ', 'ꤢꤨ', 'ꤢꤩ', 'ꤢꤪ']
    ipa = ['k', 'kʰ', 'g', 'ŋ', 's', 'sʰ', 'ʑ', 'ɲ', 't', 'tʰ', 'n', 'p', 'pʰ', 'm', 'd', 'b', 'r', 'j', 'l', 'w', 'θ', 'h', 'v', 't͡ɕ', 'a', 'ɤ', 'i', 'o', 'ɯ', 'ɛ', 'u', 'e', 'ɔ']

    return render_template('kayahli.html', pred=f'{characters[c[0]]} /{ipa[c[0]]}/')

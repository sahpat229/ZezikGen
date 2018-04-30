from flask import Flask, render_template, Markup, request, json, Response
import os
from text_generator import generate_sample, generate_real

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_text', methods=['POST'])
def generate_text():
    args = request.form.to_dict()

    dataset = args['dataset']
    n_chars_generate = int(args['n_chars_generate'])
    primer = args['primer'].strip() + ' '
    temperature = float(args['temperature'])

    sample = generate_sample(dataset, n_chars_generate, primer, temperature)
    real = generate_real(dataset, n_chars_generate)
    print(real)

    return json.dumps({
        'dataset': dataset,
        'n_chars_generate': n_chars_generate,
        'primer': primer,
        'temperature': temperature,
        'sample': sample,
        'real': real
    })

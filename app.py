from flask import Flask, render_template, Markup, request, json, Response
import os
from text_generator import generate_sample

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_text', methods=['POST'])
def generate_text():
    args = request.form.to_dict()
    sample = generate_sample(args['dataset'], int(args['n_chars_generate']), args['primer'].strip() + ' ', float(args['temperature']))
    args['sample'] = sample
    print(json.dumps(args))
    return json.dumps(args)

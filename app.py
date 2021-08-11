from flask import Flask, jsonify,  request, render_template
import joblib
import pandas as pd
from model import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        user_name = [x for x in request.form.values()]
        print(user_name)
        final_recomm = recommendation(user_name)
        graphJSON0 = final_recomm.to_json(orient='records')
        return render_template('index.html',graphJSON0=graphJSON0)
    else :
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
from flask import Flask, jsonify,  request, render_template
import joblib
import pandas as pd


app = Flask(__name__)
model_load = pd.read_pickle("./models/recommendation.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        user_name = [x for x in request.form.values()]
        print(user_name)
        output = model_load.loc[user_name[0]].sort_values(ascending=False)[0:20]
        return render_template('index.html', prediction_text='Churn Output {}'.format(output))
    else :
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
from flask import Flask, jsonify,  request, render_template
import joblib
import pandas as pd


app = Flask(__name__)
recomm_model_load = pd.read_pickle("./models/recommendation.pkl")
sentiment_model_load = pd.read_pickle("./models/sentiment.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        user_name = [x for x in request.form.values()]
        print(user_name)
        usr_recomm = recomm_model_load.loc[user_name[0]].sort_values(ascending=False)[0:20]
        usr_recomm = pd.DataFrame(usr_recomm)
        usr_recomm = usr_recomm.reset_index()
        final_recomm = usr_recomm.merge(sentiment_model_load,how='inner',left_on='index',right_on='name').sort_values(by='Perc_Pos',ascending=False)['name'][0:5]
        final_recomm = pd.DataFrame(final_recomm)
        graphJSON0 = final_recomm.to_json(orient='records')
        return render_template('index.html',graphJSON0=graphJSON0)
    else :
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
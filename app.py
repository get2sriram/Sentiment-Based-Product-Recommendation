## import the needed libraries
import json
from flask import Flask, request, render_template
from model import *

## define the app
app = Flask(__name__)

## API to send the home page to the browser
@app.route('/')
def home():
    return render_template('index.html')

## API to process the input from the web browser when "Predict" is clicked.
## If no username is provided to predict, render the home page again to the browser
## If username is provided to predict, validate the username using the validate_user function.
## If valid user then get the recommendations, format to JSON and send back to the browser for display
## Else send a message to provide a valid user
@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):

        user_name = [x for x in request.form.values()]
        print(user_name)

        user_name = user_name[0]
        if validate_user(user_name):
            final_recomm = recommendation(user_name)
            graphJSON0 = final_recomm.to_json(orient='records')
        else:
            graphJSON0 = json.dumps([{"name":"Enter a valid user to recommend top products"}])

        return render_template('index.html', graphJSON0=graphJSON0)
    else :
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
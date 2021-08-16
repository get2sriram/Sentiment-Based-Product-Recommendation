## import the need libraries
import pickle
import pandas as pd

## load the recommendation and sentiment classification model
recomm_model_load = pd.read_pickle("./models/recommendation.pkl")
sentiment_model_load = pd.read_pickle("./models/sentiment_model.pkl")

## function to validate if the input user is valid
def validate_user(usr_input):
    global recomm_model_load
    if usr_input in list(recomm_model_load.index):
        return True
    else:
        return False

## function to take the userid as input, get the top 20 recommendations for the user using the recommendation model,
## filter the top 5 products based on sentiment using the sentiment model. Please note the sentiment model prediction
## is done ahead of time and the %Positive sentiment is calculated.  This is used as the sentiment model to avoid
## processing a large number of reviews during runtime and application failures due to memory issues in heroku.
def recommendation(usr_input):
    global recomm_model_load, sentiment_model_load

    usr_recomm = recomm_model_load.loc[usr_input].sort_values(ascending=False)[0:20]
    usr_recomm = pd.DataFrame(usr_recomm)
    usr_recomm = usr_recomm.reset_index()
    final_recomm = \
    usr_recomm.merge(sentiment_model_load, how='inner', left_on='index', right_on='name').sort_values(by='Perc_Pos',
                                                                                                      ascending=False)[
        'name'][0:5]
    final_recomm = pd.DataFrame(final_recomm)
    return(final_recomm)


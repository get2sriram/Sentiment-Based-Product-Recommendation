def recommendation(usr_input):
    import pickle
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    recomm_model_load = pd.read_pickle("./models/recommendation.pkl")
    sentiment_model_load = pd.read_pickle("./models/sentiment_model.pkl")

    usr_recomm = recomm_model_load.loc[usr_input].iloc[0].sort_values(ascending=False)[0:20]
    usr_recomm = pd.DataFrame(usr_recomm)
    usr_recomm = usr_recomm.reset_index()
    final_recomm = \
    usr_recomm.merge(sentiment_model_load, how='inner', left_on='index', right_on='name').sort_values(by='Perc_Pos',
                                                                                                      ascending=False)[
        'name'][0:5]
    final_recomm = pd.DataFrame(final_recomm)
    return(final_recomm)


def recommendation(usr_input):
    import pickle
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    filename = "./models/recommendation.pkl"
    usr_recommendation_df =  pd.read_pickle(filename)

    usr_recomm = usr_recommendation_df.loc[usr_input].iloc[0].sort_values(ascending=False)[0:20]
    del usr_recommendation_df

    usr_recomm1=pd.DataFrame(usr_recomm).reset_index()
    del usr_recomm

    reviews_df = pd.read_csv("./models/preprocessed_reviews.csv")
    usr_recomm1.columns=['name','score']
    recomm_reviews_df = pd.merge(left=usr_recomm1,right=reviews_df,how='inner',on='name')
    del usr_recomm1, reviews_df

    filename = "./models/tfidf_model.pkl"
    tfidf_model = pd.read_pickle(filename)
    tfidf = tfidf_model.transform(recomm_reviews_df['review'])
    # Let's look at the dataframe
    tfidf = pd.DataFrame(tfidf.toarray(), columns = tfidf_model.get_feature_names())
    del tfidf_model
    tfidf =pd.concat([tfidf,pd.DataFrame(recomm_reviews_df['ratings']).reset_index().drop(columns='index')],axis=1)

    filename = "./models/svd.pkl"
    svd = pd.read_pickle(filename)
    tfidf_svd = svd.transform(tfidf)
    del tfidf, svd
    ## create the column names for the PCA components
    cols = ['PC'+str(i) for i in range(1,701)]

    ## concatenate the X_test PCA components with the y_test
    df_tfidf_svd = pd.DataFrame(tfidf_svd,columns = cols)

    X_tfidf = df_tfidf_svd
    del df_tfidf_svd

    filename = "./models/lr_hyp_model.pkl"
    clf_model = pd.read_pickle(filename)
    y_pred_proba = clf_model.predict_proba(X_tfidf)
    y_pred = (y_pred_proba[:,1]>0.45).astype('int')
    recomm_reviews_df['pred_sent'] = pd.Series(list(y_pred))
    del clf_model,y_pred_proba, y_pred

    sent_pivot=recomm_reviews_df.pivot_table(index='name',columns='pred_sent',values='sentiment',aggfunc='count').reset_index()
    del recomm_reviews_df
    sent_pivot.columns=['name','Positive','Negative']
    sent_pivot['Perc_Pos'] = sent_pivot['Positive']/(sent_pivot['Positive']+sent_pivot['Negative'])*100
    sent_pivot.sort_values(by='Perc_Pos',ascending=False,inplace=True)
    final_recomm = pd.DataFrame(sent_pivot['name'][:5])
    del sent_pivot

    return(final_recomm)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from joblib import load, dump

from joblib import dump, load

import features


def get_model(estimator=None):

    if estimator is not None:
        data = pd.read_csv("./data/dataset.csv", encoding="utf-8")

        tfidf_feature = pd.read_csv("./data/tfidf_feature.csv", encoding="utf-8")
        dict_feature = pd.read_csv("./data/dictionary.csv", encoding="utf-8")
        sentiment_feature = pd.read_csv("./data/sentiment.csv", encoding="utf-8")

        df_list = [data, sentiment_feature, dict_feature, dict_feature, tfidf_feature]

        master = df_list[0]

        for df in df_list[1:]:
            master = master.merge(df, on="index")

        y = master.iloc[:, 1]
        X = master.iloc[:, 3:]

        X_train, X_tv, y_train, y_tv = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        X_validate, X_test, y_validate, y_test = train_test_split(
            X_tv, y_tv, test_size=0.5, random_state=42
        )

        scaler = MinMaxScaler()
        scaler.fit(X)
        X_train_scaled = scaler.transform(X_train)
        X_validate_scaled = scaler.transform(X_validate)
        X_test_scaled = scaler.transform(X_test)

        # comp_NB = ComplementNB()
        estimator.fit(X_train_scaled, y_train)
        predictions = estimator.predict(X_test_scaled)

        print("Accuracy Score: ", accuracy_score(y_test, predictions))
        print("Confusion Matrix: \n", confusion_matrix(y_test, predictions))

    return estimator


if __name__ == "__main__":

    classifier_1 = ComplementNB()
    classifier_2 = LogisticRegression(solver="lbfgs")

    model = get_model(classifier_1)
    dump(model, "model.joblib")

    model = load("model.joblib")

    tweet = input("Tweet: ")

    while True:
        if tweet != "logout":

            sen_feature = pd.DataFrame(features.get_sentiment_score(tweet), index=[0])
            dic_feature = pd.DataFrame(features.term_frequency(tweet), index=[0])
            tfidf_feature = features.get_tfidf_scores(tweet)

            tweet_df = sen_feature.merge(dic_feature, left_index=True, right_index=True)
            tweet_df = tweet_df.merge(dic_feature, left_index=True, right_index=True)
            tweet_df = tweet_df.merge(tfidf_feature, left_index=True, right_index=True)

            pred = model.predict_proba(tweet_df)
            print("Hate Level: ", pred[0][1])
            tweet = input("Tweet: ")
        else:
            break

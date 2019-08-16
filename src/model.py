from sklearn import svm
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from joblib import dump, load

import features


def get_model():

    data = pd.read_csv('./data/dataset.csv', encoding='utf-8')
    tfidf_feature = pd.read_csv('./data/tfidf_feature.csv', encoding='utf-8')
    dict_feature = pd.read_csv('./data/dictionary.csv', encoding='utf-8')
    sentiment_feature = pd.read_csv('./data/sentiment.csv', encoding='utf-8')

    df_list = [data, sentiment_feature, dict_feature, dict_feature]

    master = df_list[0]

    for df in df_list[1:]:
        master = master.merge(df, on='index')

    y = master.iloc[:, 1]
    X = master.iloc[:, 3:]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    comp_NB = ComplementNB()
    comp_NB.fit(X_train_scaled, y_train)
    comp_pred = comp_NB.predict(X_test_scaled)

    print("Complement Naive Bayes")
    print("Accuracy Score: ", accuracy_score(y_test, comp_pred))
    print("Confusion Matrix: \n", confusion_matrix(y_test, comp_pred))

    return comp_NB


labels = pd.read_csv('./data/dataset.csv', encoding='utf-8')
dictionary = pd.read_csv('./data/dictionary.csv', encoding='utf-8')
sentiment_feature = pd.read_csv('./data/sentiment.csv', encoding='utf-8')

df_list = [labels, dictionary]


if __name__ == "__main__":

    model = get_model()
    dump(model, 'comp_NaiveBayes.joblib')

    model = load('comp_NaiveBayes.joblib')

    tweet = input("Tweet: ")

    while True:
        if tweet != "logout":
            sen_feature = pd.DataFrame(
                features.get_sentiment_score(tweet), index=[0])
            dic_feature1 = pd.DataFrame(
                features.term_frequency(tweet), index=[0])

            dic_feature2 = pd.DataFrame(
                features.term_frequency(tweet), index=[0])

            tweet_df = sen_feature.merge(
                dic_feature1, left_index=True, right_index=True)
            tweet_df = tweet_df.merge(
                dic_feature2, left_index=True, right_index=True)

            pred = model.predict_proba(tweet_df)
            print("Hate Level: ", pred[0][1])
            tweet = input("Tweet: ")
        else:
            break
for i in range(len(predictions)):
    print(predictions[i])

# Creating data for model to be trained using svm
X = pd.read_csv("data/sentiment.csv")
X = X.drop(columns=['index'])

df = pd.read_csv("data/dataset.csv")
y = df.label

# Split dataset into trainign and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)

# Train svm classifier
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

#y_pred = svclassifier.predict(X_test)

# Scale data
scaler = StandardScaler()

# Fit on training set alone
scaler.fit(X_train, y_train)

# Feature Scaling (start here)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
pca = PCA(n_components = 4)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Train Regression Model with PCA
classifier = LogisticRegression(solver = 'lbfgs')
classifier.fit(X_train, y_train)

# Predict Results from PCA Model
y_pred = classifier.predict(X_test)

# Create Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Apply transform to both the training set and the test set
# train_img = scaler.transform(X_train)
# test_img = scaler.transform(X_test)

# Making instance of the Model
# pca = PCA(.95)

# Fit pca on training set
# pca.fit(train_img)

# Apply mapping transform to both training and testing set
# train_img = pca.transform(X_train)
# test_img = pca.transform(X_test)

# In sklearn, all machine learning models are implemented as Python classes
# Making instance of model
# logisticRegr = LogisticRegression(solver = 'lbfgs')

# Training model on data
# logisticRegr.fit(train_img, y_train)

# Predict for one observation (image)
# logisticRegr.predict(test_img[0:10])

# print(confusion_matrix(y_test, y_pred))
# print(logisticRegr.score(test_img, y_pred))
# print(classification_report(y_test, y_pred))

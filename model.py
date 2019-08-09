from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import ComplementNB

labels = pd.read_csv('data/labels.csv', encoding='utf-8')
tfidf_feature = pd.read_csv('data/tfidf_feature.csv', encoding='utf-8')
dict_feature = pd.read_csv('data/dictionary.csv', encoding='utf-8')
sentiment_feature = pd.read_csv('data/sentiment.csv', encoding='utf-8')

df_list = [labels, tfidf_feature]

master = df_list[0]

for df in df_list[1:]:
    master = master.merge(df, on='index')

master.to_csv('data/feature-set.csv')

# Get the class labels for each tweet
y = master.iloc[:, 1]
# Get all the features
X = master.iloc[:, 3:]

# Training and testing datasets: 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# training set for dictionary feature
x_dic_train = pd.DataFrame(X_train.loc[:, :])
x_dic_test = pd.DataFrame(X_test.loc[:, :])

# Creating scaled version of training and testing data
scaler = StandardScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train classifier
classifier = ComplementNB()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

for i in range(len(predictions)):
    print(predictions[i])

df = pd.DataFrame(index=X_test.index)
df.to_csv('index.csv')

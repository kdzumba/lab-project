from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
#from sklearn.cross_validation import train_test_split
#from sklearn.linear_model import LogisticRegression
#import random

import Preprocessing as processor

#log_model = LogisticRegression()

def sentiment_analyser(tweet_text):
  analyser = SentimentIntensityAnalyzer()
  sentiment_score = analyser.polarity_scores(tweet_text)
  return sentiment_score

def sentiment_feature(datapath):
  data = pd.read_csv(datapath)
  dic_list = []

  for i in range(len(data)):
    dic = sentiment_analyser(data.tweet[i])
    dic_list.append(dic)

  df_list = []

  for i in range(len(dic_list)):
    df = pd.DataFrame(dic_list[i], index=[i])
    df_list.append(df)

  master = df_list[0]

  for df in df_list[1:]:
    master = pd.concat([master, df], ignore_index=True)
  
  master.to_csv('data/sentiment.csv')

# X_train, X_test, y_train, y_test  = train_test_split(features_nd, data_labels, train_size=0.80, random_state=1234)
# log_model = log_model.fit(X=X_train, y=y_train)
# y_pred = log_model.predict(X_test)

# j = random.randint(0,len(X_test)-7)
# for i in range(j,j+7):
#     print(y_pred[0])
#     ind = features_nd.tolist().index(X_test[i].tolist())
#     print(data[ind].strip())

if __name__ == '__main__':

 
  sentiment_feature('data/dataset.csv')
  #datapath = 'data/dataset.csv'


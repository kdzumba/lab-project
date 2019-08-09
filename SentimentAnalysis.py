from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()



def sentiment_analyser(tweet_text):
    sentiment_score = analyser.polarity_scores(tweet_text)
    print("{:-<40} {}".format(tweet_text, str(sentiment_score)))

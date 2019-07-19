import json

#Path to the file containing the stream of data downloaded from twitter
tweets_data_path = 'twitter_data'

#Array for storing a dictionary of tweets 
tweets_data = []

#Creating a file object for reading the tweets data 
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue

#Printing the number of tweets that have been loaded into the tweets file
print (len(tweets_data))

#Extracting a tweet text from tweets data
for tweet in tweets_data:

#Do this for tweets that have more than 140 characters, where the extended_tweet dictionary is embedded within then retweeted_status dictionary
	if 'retweeted_status' in tweet:
		if 'extended_tweet' in tweet['retweeted_status']:
			tweet_text = tweet['retweeted_status']['extended_tweet']['full_text']


#Do this for tweets that have more than 140 characters, where the extended_tweet dictionary is not embedded within the retweeted_status dictionary
	elif 'extended_tweet' in tweet:
		tweet_text = tweet['extended_tweet']['full_text']

#Do this for tweets that have 140 characters or less
	else:
		tweet_text = tweet['text']








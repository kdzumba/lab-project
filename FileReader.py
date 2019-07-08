import json
import re

#Removes all twitter usernames from the tweet text (using the regurlar expression in regx)
def remove_usernames(tweet_text):
	regex = r'@\w*\s'
	tweet_text = re.sub(regex, '', tweet_text)


tweets_data_path = 'twitter_data.txt'

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
	if 'retweeted_status' in tweet:
		if 'extended_tweet' in tweet['retweeted_status']:
			text = tweet['retweeted_status']['extended_tweet']['full_text']
			remove_usernames(text)
			print(text)
	else:
		remove_usernames(tweet['text'])
		print(text)



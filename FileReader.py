import json
import re

#Removes all twitter usernames from the tweet text (using the regurlar expression in regex)
def remove_usernames(tweet_text):
	tweet = tweet_text
	regex = r'@\w*\s' #Regurlar expression that matches twitter usernames
	match = re.search(regex,tweet_text)

	if match:
		tweet = re.sub(regex,'',tweet_text)

	return tweet

#Removes all twitter urls from the tweet text (using the regurlar expression in regxe)
def remove_urls(tweet_text):
	tweet = tweet_text
	#Regurlar expression that matches twitter urls
	regex = r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)+?([a-z0-9])+([\-\.]{1}[a-z0-9]+)*(\/)+?([A-Za-z0-9]+)'
	match = re.search(regex,tweet_text)

	if match:
		tweet = re.sub(regex,'',tweet_text)

	return tweet

#Path to the file containing the stream of data downloaded from twitter
tweets_data_path = 'twitter_data.txt'

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
	#Do this for tweets that have more than 140 characters, where the extended_tweet dictionary is embedded within then
	#retweeted_status dictionary
	if 'retweeted_status' in tweet:
		if 'extended_tweet' in tweet['retweeted_status']:
			tweet_text = tweet['retweeted_status']['extended_tweet']['full_text']
			tweet_without_username = remove_usernames(tweet_text)
			tweet_without_urls = remove_urls(tweet_without_username)
			print(tweet_without_urls)
			print('\n')
	#Do this for tweets that have more than 140 characters, where the extended_tweet dictionary is not embedded within the 
	#retweeted_status dictionary
	elif 'extended_tweet' in tweet:
		tweet_text = tweet['extended_tweet']['full_text']
		tweet_without_username = remove_usernames(tweet_text)
		tweet_without_urls = remove_urls(tweet_without_username)
		print(tweet_without_urls)
		print('here')
		print('\n')
	#Do this for tweets that have 140 characters or less
	else:
		tweet_text = tweet['text']
		tweet_without_username = remove_usernames(tweet_text)
		tweet_without_urls = remove_urls(tweet_without_username)
		print(tweet_without_urls)
		print('\n')




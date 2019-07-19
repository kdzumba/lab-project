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

#Removes all new line characters from the tweet text (using the regular expression in regex)
def remove_line(tweet_text):
	tweet = tweet_text
	regex = r'[\n]'
	match = re.search(regex, tweet_text)

	if match:
		tweet = re.sub(regex,'',tweet_text)

	return tweet

#Converts all tweets to lower case
def convert_to_lower_case(tweet_text):	
	tweet = tweet_text
	tweet = tweet.lower()
	return tweet 
	

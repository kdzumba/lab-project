"""
This file contains modules for performing various preprocessing of the raw data downloaded from Twitter.
This includes:
(1) Extracting tweet text from downloaded json tweet data
(1) Replacing urls with 'URLHERE'
(2) Replacing tags with 'TAGHERE'
(3) Replacing hashtag with 'HASHTAGHERE'
(4) Removing duplicate tweets from the downloaded data
(5) Converts all tweets to lowercase
"""

import re
import csv
import json
import pandas as pd
import string

# Removes all twitter usernames from the tweet text (using the regular expression in regex)
# and replaces them with 'TAGHERE'


def remove_usernames(tweet_text):
    tweet = tweet_text
    regex = r'@\w*\s'  # Regular expression that matches twitter usernames
    match = re.search(regex, tweet_text)

    if match:
        tweet = re.sub(regex, 'TAGHERE ', tweet_text)
    return tweet


# Removes all twitter urls from the tweet text (using the regular expression in regex)
# and replaces them with 'URLHERE'

def remove_urls(tweet_text):
    tweet = tweet_text
    # Regurlar expression that matches twitter urls
    regex = r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)+?([a-z0-9])+([\-\.]{1}[a-z0-9]+)*(\/)+?([A-Za-z0-9]+)'
    match = re.search(regex, tweet_text)

    if match:
        tweet = re.sub(regex, 'URLHERE ', tweet_text)
    return tweet


# Removes all new line characters from the tweet text (using the regular expression in regex)

def remove_line(tweet_text):
    tweet = tweet_text
    regex = r'[\n]'
    match = re.search(regex, tweet_text)

    if match:
        tweet = re.sub(regex, '', tweet_text)
    return tweet


# Converts all tweets to lower case

def convert_to_lower_case(tweet_text):
    tweet = tweet_text
    tweet = tweet.lower()
    return tweet

# Collects all hashtags in tweets

def get_all_hashtags(tweet_text):
    tweet = tweet_text
    regex = r'[#+]{1,}'
    re.findall(regex,tweet_text)
    return tweet

# Creates a csv file(output_filename) from textual data that is in input_filename

def convert_to_csv(input_filename, output_filename):
    with open(input_filename, 'r') as infile:
        with open(output_filename, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(
                ('count', 'hate_speech', 'not_hate_speech', 'class', 'tweet'))
            for text in infile:
                stripped = text.strip()
                tweet = '"' + stripped + '"'
                row = (2, ' ', ' ', ' ', tweet)
                writer.writerow(row)


# Extracts tweet text from downloaded json tweet data. The tweet data is stored
# in filename. The extracted tweets are written to a file from which output_file_object is created

def get_tweets(input_filename, output_filename):
    tweets_data = []
    with open(input_filename, 'r') as tweets_file:
        for line in tweets_file:
            try:
                tweet = json.loads(line)
                tweets_data.append(tweet)
            except:
                continue

        with open(output_filename, 'w') as output_file:
            for tweet in tweets_data:
                # Do this for tweets that have more than 140 characters, where the extended_tweet
                # dictionary is embedded within then retweeted_status dictionary
                if 'retweeted_status' in tweet:
                    if 'extended_tweet' in tweet['retweeted_status']:
                        tweet_text = tweet['retweeted_status']['extended_tweet']['full_text']
                        tweet_without_urls = remove_urls(tweet_text)
                        tweet_without_username = remove_usernames(
                            tweet_without_urls)
                        tweet_without_linebreaks = remove_line(
                            tweet_without_username)
                        output_file.write(tweet_without_linebreaks)
                        output_file.write("\n")
                # Do this for tweets that have more than 140 characters, where the extended_tweet
                # dictionary is not embedded within the retweeted_status dictionary
                elif 'extended_tweet' in tweet:
                    tweet_text = tweet['extended_tweet']['full_text']
                    tweet_without_urls = remove_urls(tweet_text)
                    tweet_without_username = remove_usernames(
                        tweet_without_urls)
                    tweet_without_linebreaks = remove_line(
                        tweet_without_username)
                    output_file.write(tweet_without_linebreaks)
                    output_file.write("\n")
                # Do this for tweets that have 140 characters or less
                else:
                    tweet_text = tweet['text']
                    tweet_without_urls = remove_urls(tweet_text)
                    tweet_without_username = remove_usernames(
                        tweet_without_urls)
                    tweet_without_linebreaks = remove_line(
                        tweet_without_username)
                    output_file.write(tweet_without_linebreaks)
                    output_file.write("\n")


# Given the path to a csv file, the function removes duplicate tweets from the file
# Only the first occurance of the tweet is retained
def remove_duplicates(filepath):
    df = pd.read_csv(filepath)
    df.drop_duplicates(subset=['tweet'], keep='first', inplace=True)
    df.to_csv(filepath, index=False)



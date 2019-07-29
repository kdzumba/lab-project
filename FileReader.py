"""
This file contains code for creating an unlabelled dataset from
raw data downloaded from Twitter. Raw data downloaded from Twitter
is saved in data/twitter_data.txt. Preprosessed data is stored in
data/dataset.txt and the overall dataset is stored in data/dataset.csv
"""
import Preprocessing as processor

#raw_data_file = "data/twitter_data.txt"
#processed_data_file = "data/dataset.txt"
dataset_file = "data/dataset.csv"

if __name__ == "__main__":

    # Get unlabelled dataset from raw tweets data
    #processor.get_tweets(raw_data_file, processed_data_file)

    # Get dataset in csv format from textfile data
    #processor.convert_to_csv(processed_data_file, dataset_file)

    # Remove duplicate tweets from the dataset
    processor.remove_duplicates(dataset_file)
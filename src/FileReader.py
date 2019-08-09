"""
This file contains code for creating an unlabelled dataset from
raw data downloaded from Twitter. Raw data downloaded from Twitter
is saved in data/twitter_data.txt. Preprosessed data is stored in
data/dataset.txt and the overall dataset is stored in data/dataset.csv
"""
import Preprocessing as processor
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# raw_data_file = "twitter_data.txt"
# processed_data_file = "data/dataset2.txt"
# dataset_file = "data/dataset2.csv"


def generate_labels(filepath):
    dataset = pd.read_csv(filepath)

    labels = pd.DataFrame(index=dataset.index)
    labels['label'] = dataset.label
    labels['tweet'] = dataset.tweet

    if 'index' in labels.columns:
        labels = labels.drop(columns=['index'])
        labels.to_csv('data/labels.csv', index=True)
    else:
        labels.to_csv(filepath, index=False)


def generate_dataset(filepath):
    df = pd.read_csv(filepath)

    if 'index' in df.columns:
        df = df.drop(columns=['index'])
        df.to_csv(filepath, index=True)
    else:
        df.to_csv(filepath, index=False)


if __name__ == "__main__":

    # # Get unlabelled dataset from raw tweets data
    # processor.get_tweets(raw_data_file, processed_data_file)

    # # # Get dataset in csv format from textfile data
    # processor.convert_to_csv(processed_data_file, dataset_file)

    # # # Remove duplicate tweets from the dataset
    # processor.remove_duplicates(dataset_file)

    # df = pd.read_csv('data/dataset.csv')

    # # # print(df.sum(skipna=True))

    # df = df.drop(columns=['index', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    # df.to_csv('dataset.csv', index=False)

    filepath = 'data/dataset.csv'

    # generate_dataset(filepath)
    generate_labels(filepath)

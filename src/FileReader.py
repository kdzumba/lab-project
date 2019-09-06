"""
This file contains code for creating an unlabelled dataset from
raw data downloaded from Twitter. Raw data downloaded from Twitter
is saved in data/twitter_data.txt. Preprosessed data is stored in
data/dataset.txt and the overall dataset is stored in data/dataset.csv
"""
import Preprocessing as processor
import pandas as pd


def generate_dataset(filepath=None):

    if filepath is not None:
        df = pd.read_csv(filepath)

        if "index" in df.columns:
            df = df.drop(columns=["index"])
            df.to_csv(filepath, index=True)
        else:
            df.to_csv(filepath, index=False)
    else:
        raw_data_file = "./data/twitter_data.txt"
        processed_data_file = "./data/tweets2.txt"
        dataset_file = "./data/unlabelled2.csv"

        processor.get_tweets(raw_data_file, processed_data_file)
        processor.convert_to_csv(processed_data_file, dataset_file)
        processor.remove_duplicates(dataset_file)


def extract_labelled(filepath):
    df = pd.read_csv(filepath)

    df.dropna(axis=0, inplace=True)
    df.to_csv("./data/labelled.csv")


if __name__ == "__main__":

    filepath = "./data/dataset.csv"
    generate_dataset(filepath)

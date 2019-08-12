"""
This file contains functions for performing various feature extraction processes.
Datasets with relevant features are created within the functions. Different 
preprocessing techniques are applied to extract different features
"""

import Preprocessing as processor
import pandas as pd
import numpy as np
import sklearn.feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
import string

from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


"""
Computes the term frequency of each of the words in a dictionary of hate speech
associated-terms and scales it by the weight associated with that term.
"""
def term_frequency(text):

    text = text.lower()
    text = processor.stem_words(text)
    words = word_tokenize(text)
    dictionary = { 
            'dumb': 0.5,
            'muthafuckin':0.5,
            'muthafucking': 0.5,
            'motherfucking': 0.5,
            'motherfucker': 0.5,
            'kill': 0.4,
            'killing':0.4,
            'moron': 0.6,
            'stupid': 0.5,
            'slaughter': 0.3,
            'massacre': 0.3,
            'genocide': 0.4,
            'criminal': 0.4,
            'thug': 0.4,
            'violent': 0.4,
            'faggot': 0.7,
            'sissy': 0.7,
            'fuck': 0.5,
            'fucking':0.5,
            'shit': 0.5,
            'disgusting': 0.6,
            'hate': 0.3,
            'baboon': 0.4,
            'coward': 0.4,
            'foreigner': 0.3,
            'outsider': 0.3,
            'ugly': 0.7,
            'uncivilised': 0.5,
            'kaffir': 0.9,
            'kaffer': 0.9,
            'kafer': 0.9,
            'kafir':0.9,
            'kuli': 0.8,
            'coolie': 0.7,
            'bitch': 0.6,
            'hoe': 0.7,
            'whore': 0.8,
            'pig': 0.5,
            'fake': 0.4,
            'fong kong': 0.4,
            'sodomy': 0.4,
            'whitey': 0.6,
            'hottentot': 0.6,
            'chinaman': 0.6,
            'chink': 0.6,
            'chinkie': 0.6,
            'coon': 0.5,
            'half breed': 0.7,
            'hairyback': 0.6,
            'jungle bunny': 0.6,
            'kraut': 0.5,
            'goy': 0.5,
            'shiksa': 0.6,
            'yid': 0.6,
            'fag': 0.7,
            'dyke': 0.6,
            'hooligan': 0.6,
            'shithole': 0.6,
            'slave': 0.3,
            'bulldyke': 0.7,
            'muppet': 0.5,
            'lunatic': 0.5,
            'shoot': 0.4,
            'pansy': 0.5,
            'terrorist': 0.4,
            'monkey': 0.3,
            'moffie': 0.7,
            'redneck': 0.5,
            'kike': 0.7,
            'abomination': 0.5,
            'lawless': 0.4,
            'amakwerekwere': 0.6,
            'kwerekwere': 0.6,
            'ofay': 0.5,
            'mohammedan': 0.5,
            'lesbo': 0.5,
            'barbaric': 0.7,
            'barbarian': 0.8,
            'ape': 0.4,
            'carpet muncher': 0.7,
            'rag muncher': 0.7,
            'bigot': 0.5,
            'your kind': 0.4,
            'your type': 0.4,
            'your people': 0.5,
            'loot': 0.4,
            'thieves': 0.4,
            'steal': 0.4,
            'dirty': 0.5,
            'garden boy': 0.6,
            'trash': 0.6,
            'garbage': 0.6,
            'honky': 0.6,
            'nigger': 0.8,
            'negro': 0.7,
            'nigga': 0.8,
            'bastard': 0.8,
            'gook': 0.6,
            'sand nigger': 0.7,
            'sand nigga': 0.7,
            'ghetto': 0.5,
            'ratchet': 0.5,
            'tranny': 0.7,
            'thot': 0.6,
            'cunt': 0.5,
            'sicko':0.5}
    columns = dictionary.keys()

    normalised = 0

    dict_features = {}

    for dic_term in dictionary.keys():
        count = 0
        for word in words:
            if word == dic_term:
                count = count + 1
        normalised = (normalised + count * dictionary[dic_term])/(count + 1)
        dict_features[dic_term] = count * dictionary[dic_term]

    dict_features['weight'] = normalised
    
    return dict_features


"""
Creates a dictionary dataset with tweets weighted based on their term frequency
for words in the dictionary of hate speech associable words.
"""
def dictionary_feature(datapath):

    data = pd.read_csv(datapath)

    master_df = pd.DataFrame(term_frequency(data.tweet[0]), index=[0])

    for i in range(1, len(data)):
        master_df = pd.concat([master_df, pd.DataFrame(term_frequency(data.tweet[i]), index=[i])], ignore_index=True)
    
    master_df = master_df.to_dense()
    master_df.to_csv('./data/dictionary.csv')


"""
Calculates the sentiment score for the string in text
This needs to be done before:
(1) Converting to lowercase
(2) Removing emojis
(3) Remomving puncutations
(4) Removing emoticons
(5) Removing stopwords
This is because vaderSentiment analyzer uses the above metioned properties of 
text to compute a more representative sentiment score
"""
def get_sentiment_score(text):

    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(text)
    return score


"""
Makes use of the get_sentiment_score function to create a dataset with sentiment
features
"""
def sentiment_feature(datapath):
    
    data = pd.read_csv(datapath)
    dic_list = []

    for i in range(len(data)):
        dic = get_sentiment_score(data.tweet[i])
        dic_list.append(dic)

    df_list = []

    for i in range(len(dic_list)):
        df = pd.DataFrame(dic_list[i], index=[i])
        df_list.append(df)

    master = df_list[0]

    for df in df_list[1:]:
        master = pd.concat([master, df], ignore_index=True)

    master.to_csv('./data/sentiment.csv')

def normalise_data(datapath):
    
    data = pd.read_csv(datapath)

    df = pd.DataFrame()

    #normalise 'compound' column in sentiment.csv file
    unnormalised_data = np.array(df['compound'])
    normalised_data = preprocessing.normalize([unnormalised_data])

    sentiment_feature('./data/sentiment.csv')

    master = df_list[0]

    master.to_csv('./data/normalisedsentiment.csv')

"""
calculates TF-IDF scores for anagrams, bigrams, trigrams and quadgrams from the 
dataset in filepath and generates a tfidf feature dataset. Does this after removing
stopwords, punctuations and digits because tf-idf scores don't factor in sentence
structure, hence including them would be wasteful
"""
def tfidf_feature(datapath):

    data = pd.read_csv(datapath)

    df = pd.DataFrame()

    # preprocessing required for tf_idf feature
    df['tweet'] = data.tweet
    # df['tweet'] = df.tweet.map(processor.remove_stopwords)
    # df['tweet'] = df.tweet.map(processor.remove_punctuations)
    df['tweet'] = df.tweet.map(processor.remove_digits)

    # Convert corpus into a bag of words. Each comment is converted to a vector
    # of words (semantic structure ignored)
    count_v = CountVectorizer(ngram_range=(2, 4), min_df=0.001)

    # Count the number of times each word occurs on a comment
    count_v.fit(df.tweet)
    count_matrix = count_v.transform(df.tweet)

    # Compute tfidf weights for preprocessed comments
    transformer = TfidfTransformer()
    tf_idf_weights = transformer.fit_transform(count_matrix)

    tf_idf = pd.DataFrame(tf_idf_weights.todense(
    ), index=data['index'], columns=count_v.get_feature_names())
    tf_idf.to_csv('./data/tfidf_feature.csv')


if __name__ == '__main__':

    datapath = 'data/dataset.csv'

    # dictionary_feature(datapath)
    sentiment_feature(datapath)
    # tfidf_feature(datapath)

    text = 'dirty'
    stem = processor.stem_words(text)
    print(stem)
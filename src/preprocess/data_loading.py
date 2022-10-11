import numpy as np
import pandas as pd

from consts import IMDB_LABEL_MAPPINGS, YELP_REVIEW_MAPPING


def load_raw_imdb(path):
    df = pd.read_csv(path)
    df['sentiment'] = df['sentiment'].apply(lambda sentiment: IMDB_LABEL_MAPPINGS[sentiment])
    df = df.dropna()
    return df


def load_raw_corona_data(path):
    df = pd.read_csv(path, encoding='latin-1', usecols=['OriginalTweet', 'Sentiment'])
    df = df.rename(columns={'OriginalTweet': 'text', 'Sentiment': 'sentiment'})
    df = df.dropna()
    return df


def load_yelp_data(path):
    df = pd.read_csv(path, names=['sentiment', 'text'])
    df['sentiment'] = df['sentiment'].apply(lambda sentiment: YELP_REVIEW_MAPPING[sentiment])
    # reorder columns, so first it goes text, then sentiment, as in other datasets.
    cols = df.columns.tolist()
    cols = [cols[1], cols[0]]
    df = df.dropna()
    return df[cols]


def load_preprocessed_data(path):
    df = pd.read_csv(path)
    return df


def get_data(path, column_text, label_mapping):
    data = load_preprocessed_data(path)[[column_text, 'sentiment']]
    # shuffle the dataframe
    data = data.sample(frac=1)
    # sometimes original text is not empty, but after preprocessing, stemming/lemmatization can be, so we need to drop
    # those columns
    data = data.dropna()
    X = data[column_text]
    y = np.array(data['sentiment'])
    y = np.array([label_mapping[label] for label in y])
    return X, y

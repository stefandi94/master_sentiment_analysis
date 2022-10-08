import re

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import preprocessor as p

from src.utils import create_dir_path
from src.preprocess import load_raw_imdb, load_raw_corona_data, load_yelp_data
from consts import *

porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()


class Preprocess:
    def __init__(self):
        self.porter_stemmer = PorterStemmer()
        self.wordnet_lemmatizer = WordNetLemmatizer()

    @staticmethod
    def clean_text(text):
        text = re.sub('[^A-Za-z0-9]+', ' ', text)
        text = " ".join(text.split())
        text = text.lower().strip()
        return text

    def clean_tweets(self, text):
        text = p.clean(text)
        text = self.clean_text(text)
        return text

    @staticmethod
    def tokenization(text):
        tokenized_text = word_tokenize(text)
        tokenized_text = [word for word in tokenized_text]
        return tokenized_text

    @staticmethod
    def stemming(tokenized_text):
        tokenized_stem_text = [porter_stemmer.stem(word) for word in tokenized_text]
        return tokenized_stem_text

    @staticmethod
    def lemmatizer(tokenized_text):
        tokenized_lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in tokenized_text]
        return tokenized_lemm_text


def one_iter_preprocess(df, twitter):
    clean_text_df = []
    tokenized_clean_text_df = []
    stem_clean_text_df = []
    tokenized_stem_clean_text_df = []
    lemm_clean_text_df = []
    tokenized_lemm_clean_text_df = []

    text_df = list(df['text'].values)
    if twitter:
        print(f'Cleaning tweets!')
        preprocessing = preprocess.clean_tweets
    else:
        preprocessing = preprocess.clean_text

    for text in tqdm(text_df, desc="Row loop"):
        clean_text = preprocessing(text)
        clean_text_df.append(clean_text)

        tokenized_clean_text = preprocess.tokenization(clean_text)
        tokenized_clean_text_df.append(tokenized_clean_text)

        tokenized_stem_text = preprocess.stemming(tokenized_clean_text)
        tokenized_stem_clean_text_df.append(tokenized_stem_text)
        stem_clean_text_df.append(" ".join(tokenized_stem_text))

        tokenize_lemm_clean_text = preprocess.lemmatizer(tokenized_clean_text)
        tokenized_lemm_clean_text_df.append(tokenize_lemm_clean_text)
        lemm_clean_text_df.append(" ".join(tokenize_lemm_clean_text))

    return clean_text_df, tokenized_clean_text_df, stem_clean_text_df, tokenized_stem_clean_text_df, lemm_clean_text_df, tokenized_lemm_clean_text_df


def data_preprocess(df, twitter):
    clean_text_df, tokenized_clean_text_df, stem_clean_text_df, tokenized_stem_clean_text_df, lemm_clean_text_df, tokenized_lemm_clean_text_df = one_iter_preprocess(df, twitter=twitter)
    df['clean_text'] = clean_text_df
    df['tokenized_clean_text'] = tokenized_clean_text_df
    df['tokenized_stem_clean_text'] = tokenized_stem_clean_text_df
    df['stem_clean_text'] = stem_clean_text_df
    df['tokenized_lemm_clean_text'] = tokenized_lemm_clean_text_df
    df['lemm_clean_text'] = lemm_clean_text_df
    return df


def preprocess_and_save(df, file_path, twitter):
    df = data_preprocess(df, twitter=twitter)
    df = df.dropna()
    create_dir_path(file_path)
    df.to_csv(file_path, index=False)


def transform_and_save(load_function, raw_path, preprocessed_path, twitter=False):
    df = load_function(raw_path)
    preprocess_and_save(df, preprocessed_path, twitter=twitter)


def transform_imdb():
    transform_and_save(load_raw_imdb, IMDB_MOVIE_REVIEWS_RAW_TRAIN, IMDB_MOVIE_REVIEWS_PROCESSED_TRAIN)
    transform_and_save(load_raw_imdb, IMDB_MOVIE_REVIEWS_RAW_TEST, IMDB_MOVIE_REVIEWS_PROCESSED_TEST)


def transform_yelp():
    transform_and_save(load_yelp_data, YELP_REVIEW_RAW_TRAIN, YELP_REVIEW_PROCESSED_TRAIN)
    transform_and_save(load_yelp_data, YELP_REVIEW_RAW_TEST, YELP_REVIEW_PROCESSED_TEST)


def transform_corona():
    transform_and_save(load_raw_corona_data, CORONA_TWEETS_DATASET_RAW_TRAIN, CORONA_TWEETS_DATASET_PROCESSED_TRAIN, twitter=True)
    transform_and_save(load_raw_corona_data, CORONA_TWEETS_DATASET_RAW_TEST, CORONA_TWEETS_DATASET_PROCESSED_TEST, twitter=True)


def transform_all():
    print(f'Transforming IMDB!')
    transform_imdb()

    print(f'Transforming YELP!')
    transform_yelp()

    print(f'Transforming Corona!')
    transform_corona()


if __name__ == '__main__':
    preprocess = Preprocess()
    transform_all()


import unittest

import pandas as pd

from consts import IMDB_LABELS, YELP_REVIEW_LABELS, IMDB_MOVIE_REVIEWS_RAW_TRAIN, YELP_REVIEW_RAW_TRAIN, \
    CORONA_TWEETS_DATASET_RAW_TRAIN, CORONA_LABELS
from src.preprocess import load_raw_imdb, load_yelp_data, load_raw_corona_data

COLUMNS = ['text', 'sentiment']


class TestImdbRawData(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.df: pd.DataFrame = load_raw_imdb(IMDB_MOVIE_REVIEWS_RAW_TRAIN)

    def test_column_names(self):
        assert COLUMNS == self.df.columns.to_list()

    def test_text_type(self):
        assert all([(type(text)) == str for text in self.df['text']])

    def test_labels(self):
        assert all([sentiment in IMDB_LABELS for sentiment in self.df['sentiment']])

    def test_shape(self):
        assert self.df.shape == (25000, len(COLUMNS))


class TestTwitterRawData(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = load_yelp_data(YELP_REVIEW_RAW_TRAIN)

    def test_column_names(self):
        assert COLUMNS == self.df.columns.to_list()

    def test_text_type(self):
        assert all([(type(text)) == str for text in self.df['text']])

    def test_labels(self):
        assert all([sentiment in YELP_REVIEW_LABELS for sentiment in self.df['sentiment']])

    def test_shape(self):
        assert self.df.shape == (40001, len(COLUMNS))


class TestCoronaRawData(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = load_raw_corona_data(CORONA_TWEETS_DATASET_RAW_TRAIN)

    def test_column_names(self):
        assert COLUMNS == self.df.columns.to_list()

    def test_text_type(self):
        assert all([(type(text)) == str for text in self.df['text']])

    def test_labels(self):
        assert all([sentiment in CORONA_LABELS for sentiment in self.df['sentiment']])

    def test_shape(self):
        assert self.df.shape == (41157, len(COLUMNS))


if __name__ == '__main__':
    unittest.main()

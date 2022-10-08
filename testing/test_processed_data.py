import unittest

import pandas as pd

from consts import YELP_REVIEW_LABELS, YELP_REVIEW_PROCESSED_TRAIN, IMDB_MOVIE_REVIEWS_PROCESSED_TRAIN, IMDB_LABELS, \
    CORONA_TWEETS_DATASET_PROCESSED_TRAIN, CORONA_LABELS
from src.preprocess.data_loading import load_preprocessed_data

COLUMNS = [
    'text', 'sentiment', 'clean_text', 'stem_clean_text', 'lemm_clean_text'
]


class TestImdbProcessedData(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.df: pd.DataFrame = load_preprocessed_data(IMDB_MOVIE_REVIEWS_PROCESSED_TRAIN)

    def test_column_names(self):
        assert COLUMNS == self.df.columns.to_list()

    def test_text_type(self):
        # not all columns are covered here, some preprocessed columns can be null, but they will be dropped
        assert all([(type(text)) == str for text in self.df['text']])

    def test_labels(self):
        assert all([sentiment in IMDB_LABELS for sentiment in self.df['sentiment']])

    def test_shape(self):
        assert self.df.shape == (25000, len(COLUMNS))


class TestYelpProcessedData(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = load_preprocessed_data(YELP_REVIEW_PROCESSED_TRAIN)

    def test_column_names(self):
        assert COLUMNS == self.df.columns.to_list()

    def test_text_type(self):
        # not all columns are covered here, some preprocessed columns can be null, but they will be dropped
        assert all([(type(text)) == str for text in self.df['text']])

    def test_labels(self):
        assert all([sentiment in YELP_REVIEW_LABELS for sentiment in self.df['sentiment']])

    def test_shape(self):
        assert self.df.shape == (40001, len(COLUMNS))


class TestCoronaProcessedData(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = load_preprocessed_data(CORONA_TWEETS_DATASET_PROCESSED_TRAIN)

    def test_column_names(self):
        assert COLUMNS == self.df.columns.to_list()

    def test_text_type(self):
        # not all columns are covered here, some preprocessed columns can be null, but they will be dropped
        assert all([(type(text)) == str for text in self.df['text']])

    def test_labels(self):
        assert all([sentiment in CORONA_LABELS for sentiment in self.df['sentiment']])

    def test_shape(self):
        assert self.df.shape == (41157, len(COLUMNS))


if __name__ == '__main__':
    unittest.main()

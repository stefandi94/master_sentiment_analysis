import os

BASE_DIR = os.path.join(os.path.dirname(__file__))

LOG_DIR = os.path.join(BASE_DIR, "logs")

##############################   DATA   ################################################################################

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA = os.path.join(DATA_DIR, "raw")
PREPROCESSED_DATA = os.path.join(DATA_DIR, "preprocessed")

###########################  RAW DATA PATHS  ###########################################################################

CORONA_TWEETS_DATASET_RAW_DIR = os.path.join(RAW_DATA, "corona_tweets")
YELP_REVIEW_RAW_DIR = os.path.join(RAW_DATA, "yelp_review")
IMDB_MOVIE_REVIEWS_RAW_DIR = os.path.join(RAW_DATA, "imdb_50k_movie_reviews")

CORONA_TWEETS_DATASET_RAW_TRAIN = os.path.join(CORONA_TWEETS_DATASET_RAW_DIR, "CORONA_NLP_train.csv")
CORONA_TWEETS_DATASET_RAW_TEST = os.path.join(CORONA_TWEETS_DATASET_RAW_DIR, "CORONA_NLP_test.csv")

YELP_REVIEW_RAW_TRAIN = os.path.join(YELP_REVIEW_RAW_DIR, "train_sample.csv")
YELP_REVIEW_RAW_TEST = os.path.join(YELP_REVIEW_RAW_DIR, "test.csv")

IMDB_MOVIE_REVIEWS_RAW_TRAIN = os.path.join(IMDB_MOVIE_REVIEWS_RAW_DIR, "train.csv")
IMDB_MOVIE_REVIEWS_RAW_TEST = os.path.join(IMDB_MOVIE_REVIEWS_RAW_DIR, "test.csv")

##########################  PROCESSED DATA PATHS  ######################################################################

CORONA_TWEETS_DATASET_PROCESSED_DIR = os.path.join(PREPROCESSED_DATA, "corona_tweets")
YELP_REVIEW_PROCESSED_DIR = os.path.join(PREPROCESSED_DATA, "yelp_review")
IMDB_MOVIE_REVIEWS_PROCESSED_DIR = os.path.join(PREPROCESSED_DATA, "imdb_50k_movie_reviews")

CORONA_TWEETS_DATASET_PROCESSED_TRAIN = os.path.join(CORONA_TWEETS_DATASET_PROCESSED_DIR, "Corona_NLP_train.csv")
CORONA_TWEETS_DATASET_PROCESSED_TEST = os.path.join(CORONA_TWEETS_DATASET_PROCESSED_DIR, "Corona_NLP_test.csv")

YELP_REVIEW_PROCESSED_TRAIN = os.path.join(YELP_REVIEW_PROCESSED_DIR, "train_sample.csv")
YELP_REVIEW_PROCESSED_TEST = os.path.join(YELP_REVIEW_PROCESSED_DIR, "test.csv")

IMDB_MOVIE_REVIEWS_PROCESSED_TRAIN = os.path.join(IMDB_MOVIE_REVIEWS_PROCESSED_DIR, "train.csv")
IMDB_MOVIE_REVIEWS_PROCESSED_TEST = os.path.join(IMDB_MOVIE_REVIEWS_PROCESSED_DIR, "test.csv")

DATASET_PATHS = {
    "imdb": IMDB_MOVIE_REVIEWS_PROCESSED_TRAIN,
    "yelp": YELP_REVIEW_PROCESSED_TRAIN,
    "corona": CORONA_TWEETS_DATASET_PROCESSED_TRAIN
}

#########################  RESULTS DIR  ################################################################################

RESULTS_DIR = os.path.join(BASE_DIR, "results")

##########################  CLASSIFICATION MODELS PATH  ################################################################

CLASSIFICATION_MODELS_DIR = os.path.join(BASE_DIR, "models")

##############################   EMBEDDINGS   ##########################################################################

EMBEDDINGS_MODELS_DIR = os.path.join(BASE_DIR, "embeddings")

#############################  PRETRAINED WORD EMBEDDINGS  #############################################################

PRETRAINED_WORD_EMBEDDINGS_DIR = os.path.join(EMBEDDINGS_MODELS_DIR, "pretrained_word_embeddings")

WORD2VEC_DIR = os.path.join(PRETRAINED_WORD_EMBEDDINGS_DIR, "word2vec")
GLOVE_DIR = os.path.join(PRETRAINED_WORD_EMBEDDINGS_DIR, "glove")
FASTTEXT_DIR = os.path.join(PRETRAINED_WORD_EMBEDDINGS_DIR, "fasttext")

GENSIM_MODEL_NAME_TO_PATH = {
    "glove-wiki-gigaword-300": GLOVE_DIR,
    "fasttext-wiki-news-subwords-300": FASTTEXT_DIR,
    "word2vec-google-news-300": WORD2VEC_DIR
}
GENSIM_MODEL_PATHS = {
    "glove": os.path.join(GLOVE_DIR, "glove-wiki-gigaword-300.kv"),
    "fasttext": os.path.join(FASTTEXT_DIR, "fasttext-wiki-news-subwords-300.kv"),
    "word2vec": os.path.join(WORD2VEC_DIR, "word2vec-google-news-300.kv"),
}

##########################  CUSTOM WORD EMBEDDINGS  ####################################################################
MAX_VOCAB_SIZE = 60000

CUSTOM_WORD_EMBEDDINGS_DIR = os.path.join(EMBEDDINGS_MODELS_DIR, "custom_word_embeddings")

IMDB_VOCAB_PATH = os.path.join(CUSTOM_WORD_EMBEDDINGS_DIR, "imdb")
IMDB_VOCAB_CLEAN_TEXT = os.path.join(IMDB_VOCAB_PATH, "tokenized_clean_text", "vocab.json")
IMDB_VOCAB_STEM_TEXT = os.path.join(IMDB_VOCAB_PATH, "tokenized_stem_clean_text", "vocab.json")
IMDB_VOCAB_LEMM_TEXT = os.path.join(IMDB_VOCAB_PATH, "tokenized_lemm_clean_text", "vocab.json")

CORONA_VOCAB_PATH = os.path.join(CUSTOM_WORD_EMBEDDINGS_DIR, "corona")
CORONA_VOCAB_CLEAN_TEXT = os.path.join(CORONA_VOCAB_PATH, "tokenized_clean_text", "vocab.json")
CORONA_VOCAB_STEM_TEXT = os.path.join(CORONA_VOCAB_PATH, "tokenized_stem_clean_text", "vocab.json")
CORONA_VOCAB_LEMM_TEXT = os.path.join(CORONA_VOCAB_PATH, "tokenized_lemm_clean_text", "vocab.json")

YELP_VOCAB_PATH = os.path.join(CUSTOM_WORD_EMBEDDINGS_DIR, "yelp")
YELP_VOCAB_CLEAN_TEXT = os.path.join(YELP_VOCAB_PATH, "tokenized_clean_text", "vocab.json")
YELP_VOCAB_STEM_TEXT = os.path.join(YELP_VOCAB_PATH, "tokenized_stem_clean_text", "vocab.json")
YELP_VOCAB_LEMM_TEXT = os.path.join(YELP_VOCAB_PATH, "tokenized_lemm_clean_text", "vocab.json")

CUSTOM_WORD_EMBEDDING_DICT_PATHS = {
    "imdb":
        {
            "tokenized_clean_text": IMDB_VOCAB_CLEAN_TEXT,
            "tokenized_stem_clean_text": IMDB_VOCAB_STEM_TEXT,
            "tokenized_lemm_clean_text": IMDB_VOCAB_LEMM_TEXT
        },
    "yelp":
        {
            "tokenized_clean_text": YELP_VOCAB_CLEAN_TEXT,
            "tokenized_stem_clean_text": YELP_VOCAB_STEM_TEXT,
            "tokenized_lemm_clean_text": YELP_VOCAB_LEMM_TEXT
        },
    "corona":
        {
            "tokenized_clean_text": CORONA_VOCAB_CLEAN_TEXT,
            "tokenized_stem_clean_text": CORONA_VOCAB_STEM_TEXT,
            "tokenized_lemm_clean_text": CORONA_VOCAB_LEMM_TEXT
        },
}
##########################  REGEX  #####################################################################################

RE_CLEAN = r"!”#$%&'()*+,-./:;?@[\]^_`{|}~"

##########################  LABELS  ####################################################################################

CORONA_LABELS = ['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive']
YELP_REVIEW_MAPPING = {1: 'Extremely Negative', 2: 'Negative', 3: 'Neutral', 4: 'Positive', 5: 'Extremely Positive'}
YELP_REVIEW_LABELS = list(YELP_REVIEW_MAPPING.values())
IMDB_LABEL_MAPPINGS = {'neg': 'Negative', 'pos': 'Positive'}
IMDB_LABELS = list(IMDB_LABEL_MAPPINGS.values())

YELP_LABEL_TO_INDEX = {'Extremely Negative': 0, 'Negative': 1, 'Neutral': 2, 'Positive': 3, 'Extremely Positive': 4}
CORONA_LABEL_TO_INDEX = {'Extremely Negative': 0, 'Negative': 1, 'Neutral': 2, 'Positive': 3, 'Extremely Positive': 4}
IMDB_LABEL_TO_INDEX = {'Negative': 0, 'Positive': 1}

DATASET_LABEL_TO_INDEX = {
    "yelp": YELP_LABEL_TO_INDEX,
    "corona": CORONA_LABEL_TO_INDEX,
    "imdb": IMDB_LABEL_TO_INDEX
}

ALLOWED_MODEL_NAMES = ["nb", "lr", "svm", "rf", "mlp"]
ALLOWED_TEXT_COLUMNS = ["clean_text", "stem_clean_text", "lemm_clean_text"]
ALLOWED_TOKENIZED_TEXT_COLUMNS = ["tokenized_clean_text", "tokenized_stem_clean_text", "tokenized_lemm_clean_text"]

MAX_SEQ_LEN = 512

CV = 10
import ast

import numpy as np
from gensim.models import KeyedVectors
from nltk import word_tokenize

from consts import GENSIM_MODEL_PATHS


def insert_oov(gensim_model):
    gensim_model.most_similar("the")
    gensim_model.add_vector(key="oov", vector=np.zeros(300, ))
    gensim_model.norms = np.append(gensim_model.norms, 0.001)
    return gensim_model


def load_gensim_embeddings(model_name):
    if model_name not in GENSIM_MODEL_PATHS.keys():
        raise TypeError(
            f'Chosen model: {model_name} is invalid! Please choose from `glove`, `fasttext` and `word2vec`.')

    model = KeyedVectors.load(GENSIM_MODEL_PATHS[model_name])
    model = insert_oov(model)
    return model


def get_gensim_word_index(gensim_model, word):
    try:
        return gensim_model.key_to_index[word]
    except KeyError:
        return gensim_model.key_to_index['oov']


def get_gensim_sentence_indices(gensim_model, sentence):
    return [get_gensim_word_index(gensim_model, word) for word in sentence]


def get_gensim_sentences_indices(gensim_model, sentences):
    return [get_gensim_sentence_indices(gensim_model, ast.literal_eval(sentence)) for sentence in sentences]


def get_embeddings(embedding, text):
    try:
        vector = embedding.get_vector(text)
    except KeyError:
        vector = np.zeros(300)
    return vector


def tokenize_text(sentence):
    return word_tokenize(sentence)


def get_sentence_embedding(embedding, sentence):
    sent_vec = [get_embeddings(embedding, word) for word in sentence]
    return sent_vec


def get_word_embeddings(X, y, embedding):
    X = [tokenize_text(x) for x in X]
    X_vec = [np.mean(get_sentence_embedding(embedding, x), axis=0) for x in X]

    X_vec = np.array(X_vec)
    y = np.array(y)
    return X_vec, y

import ast
from collections import Counter


def get_words(tokenized_sentences):
    all_words = [token for tokenized_sentence in tokenized_sentences for token in ast.literal_eval(tokenized_sentence)]
    return all_words


def filter_words(all_words, vocab_size):
    c = Counter(all_words).most_common(vocab_size - 1)
    return [element[0] for element in c]


def build_vocab(sentences, vocab_size):
    all_words = get_words(sentences)
    vocab = filter_words(all_words, vocab_size)
    # leave index 0 for padded elements
    return dict((word, index + 2) for index, word in enumerate(vocab))


def get_word_index(word, vocab):
    return vocab.get(word, 1)


def transform(sentences, vocab):
    return [[get_word_index(word, vocab) for word in ast.literal_eval(sentence)] for sentence in sentences]

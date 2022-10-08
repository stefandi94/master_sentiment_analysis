import torch
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.dl.datasets import BertEmbeddingDataset, PretrainedWordEmbeddingsDataset, CustomWordEmbeddingsDataset
from src.dl.models import SentimentCustomWordEmbeddingModel, SentimentPretrainedWordEmbeddingModel, SentimentBertEmbeddingModel


def collate_batch(batch):
    label_list, text_list, = [], []

    for (_text, _label) in batch:
        label_list.append(_label)
        text_list.append(torch.Tensor(_text))

    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)

    return text_list, label_list,


def calc_metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=None)
    norm_cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='all')
    recall = recall_score(y_true, y_pred, average="micro")
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="micro")
    f1 = f1_score(y_true, y_pred, average="micro")
    return {"f1": f1, "precision": precision, "recall": recall, "accuracy": acc}


def get_model(model_name, **kwargs):
    if model_name == "custom_embeddings":
        model = SentimentCustomWordEmbeddingModel(**kwargs)
    elif model_name == "word_embeddings":
        model = SentimentPretrainedWordEmbeddingModel(**kwargs)
    elif model_name == "bert_embeddings":
        model = SentimentBertEmbeddingModel(**kwargs)
    else:
        raise Exception(f'Model: {model_name} not available! '
                        f'Please choose from: `custom_embeddings`, `word_embeddings` and `bert_embeddings')
    return model


def get_custom_word_embedding_train_data_loader(X_train, y_train, vocab, batch_size):
    training_set = CustomWordEmbeddingsDataset(X_train, y_train, vocab)
    data_loader_train = DataLoader(batch_size=batch_size, dataset=training_set, num_workers=8, collate_fn=collate_batch)
    return data_loader_train


def get_custom_word_embedding_valid_data_loader(X_valid, y_valid, vocab, batch_size):
    valid_set = CustomWordEmbeddingsDataset(X_valid, y_valid, vocab)
    data_loader_valid = DataLoader(valid_set, collate_fn=collate_batch, batch_size=batch_size)
    return data_loader_valid


def get_pretrained_word_embedding_data_loader(X, y, word_embedding_model, batch_size):
    dataset = PretrainedWordEmbeddingsDataset(X, y, word_embedding_model)
    data_loader = DataLoader(batch_size=batch_size, dataset=dataset, num_workers=8, collate_fn=collate_batch)
    return data_loader


def get_bert_embedding_data_loader(X, y, tokenizer, batch_size):
    encodings = tokenizer(X, return_tensors="pt", truncation=True, padding=True)['input_ids']
    dataset = BertEmbeddingDataset(encodings, y)
    data_loader = DataLoader(batch_size=batch_size, dataset=dataset, num_workers=8)
    return data_loader
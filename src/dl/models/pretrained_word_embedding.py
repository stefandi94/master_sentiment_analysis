import torch
from torch import nn


class SentimentWordEmbeddingModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim


class SentimentPretrainedWordEmbeddingModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, embedding, dropout, bidirectional, freeze, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding.vectors), freeze=freeze)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        if bidirectional:
            hidden_dim *= 2
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text):
        embedding = self.embedding(text)

        lstm_out, _ = self.lstm(embedding)
        avg_pool = torch.mean(lstm_out, 1)

        output = self.dropout(avg_pool)
        output = self.fc(output)
        output = self.softmax(output)
        return output

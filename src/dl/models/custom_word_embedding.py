import torch
from torch import nn


class SentimentCustomWordEmbeddingModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, dropout, bidirectional):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
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


if __name__ == '__main__':
    print()

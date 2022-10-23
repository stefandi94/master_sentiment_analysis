import torch
from torch import nn


class SentimentCustomWordEmbeddingModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, dropout, bidirectional, last_feature=True, avg_pool=True, max_pool=True):
        super().__init__()

        assert all([last_feature, avg_pool, max_pool]), "At least one LSTM output should be used!"
        c = sum([last_feature, avg_pool, max_pool])
        self.last_feature = last_feature
        self.avg_pool = avg_pool
        self.max_pool = max_pool

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        if bidirectional:
            hidden_dim *= 2
        self.fc = nn.Linear(c * hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text):
        dense_input = []
        embedding = self.embedding(text)

        lstm_out, (hidden_state, cell_state) = self.lstm(embedding)
        if self.avg_pool:
            dense_input.append(torch.mean(lstm_out, 1))
        if self.max_pool:
            dense_input.append(torch.max(lstm_out, 1).values)
        if self.last_feature:
            dense_input.append(lstm_out[:, -1])

        conc = torch.cat(dense_input, 1)

        output = self.dropout(conc)
        output = self.fc(output)
        output = self.softmax(output)
        return output


if __name__ == '__main__':
    print()

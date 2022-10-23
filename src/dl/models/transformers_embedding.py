import torch
from torch import nn


class SentimentTransformerEmbeddingModel(nn.Module):
    def __init__(self, embeddings, hidden_dim, output_dim, bidirectional, dropout, last_feature=True, avg_pool=True, max_pool=True):
        super().__init__()

        assert all([last_feature, avg_pool, max_pool]), "At least one LSTM output should be used!"
        c = sum([last_feature, avg_pool, max_pool])
        self.embeddings = embeddings
        self.last_feature = last_feature
        self.avg_pool = avg_pool
        self.max_pool = max_pool

        embedding_dim = embeddings.embedding_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        if bidirectional:
            hidden_dim *= 2
        self.fc = nn.Linear(c * hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, indices):
        dense_input = []
        with torch.no_grad():
            embedding = self.embeddings(indices)

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

import torch
from torch import nn


class SentimentTransformerEmbeddingModel(nn.Module):
    def __init__(self, transformer_model, hidden_dim, output_dim, bidirectional, dropout):
        super().__init__()

        self.transformer_model = transformer_model
        embedding_dim = self.transformer_model.config.to_dict()['hidden_size']
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        if bidirectional:
            hidden_dim *= 2
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, indices):
        with torch.no_grad():
            embedding = self.transformer_model(indices)[0]

        lstm_out, _ = self.lstm(embedding)
        avg_pool = torch.mean(lstm_out, 1)

        output = self.dropout(avg_pool)
        output = self.fc(output)
        output = self.softmax(output)
        return output

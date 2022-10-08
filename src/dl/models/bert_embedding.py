import torch
from torch import nn
from transformers import BertModel


class SentimentBertEmbeddingModel(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, indices):
        with torch.no_grad():
            embedding = self.bert(indices)[0]

        lstm_out, _ = self.lstm(embedding)
        avg_pool = torch.mean(lstm_out, 1)

        output = self.fc(avg_pool)
        output = self.softmax(output)
        return output
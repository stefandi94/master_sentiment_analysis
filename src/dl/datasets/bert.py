from torch.utils.data import Dataset


class BertEmbeddingDataset(Dataset):
    def __init__(self, X, labels):
        self.X = X
        self.labels = labels

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

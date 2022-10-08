from torch.utils.data import Dataset

from consts import MAX_SEQ_LEN
from src.embeddings.custom_embedding import transform


class CustomWordEmbeddingsDataset(Dataset):
    def __init__(self, X, y, vocab):
        self.X = X
        self.y = y
        self.vocab = vocab
        self.words_to_ints()

    def words_to_ints(self):
        self.X = transform(self.X, self.vocab)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx][:MAX_SEQ_LEN], self.y[idx]
from torch.utils.data import Dataset

from consts import MAX_SEQ_LEN
from src.embeddings.pretrained_embedding import get_gensim_sentences_indices


class PretrainedWordEmbeddingsDataset(Dataset):
    def __init__(self, X, y, gensim_model):
        self.X = X
        self.y = y
        self.gensim_model = gensim_model

        self.words_to_ints()

    def words_to_ints(self):
        self.X = get_gensim_sentences_indices(self.gensim_model, self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx][:MAX_SEQ_LEN], self.y[idx]
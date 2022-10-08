from consts import MAX_VOCAB_SIZE
from src.dl.trainers import BaseTrainer
from src.dl.utils import get_custom_word_embedding_train_data_loader, get_custom_word_embedding_valid_data_loader, calc_metrics
from src.embeddings.custom_embedding import build_vocab


class CustomWordEmbeddingTrainer(BaseTrainer):
    def train(self, X_train, y_train, X_valid, y_valid):

        vocab = build_vocab(X_train, MAX_VOCAB_SIZE)
        train_loader = get_custom_word_embedding_train_data_loader(X_train, y_train, vocab, self.batch_size)
        valid_loader = get_custom_word_embedding_valid_data_loader(X_valid, y_valid, vocab, self.batch_size)

        best_val_f1 = 0
        best_metrics = {}
        for epoch in range(1, self.epochs + 1):
            print(f'Epoch: {epoch}')
            self.train_loop(self.model, self.optimizer, train_loader)
            valid_true, valid_predictions, test_loss = self.eval_loop(self.model, valid_loader)

            self.scheduler.step()

            valid_metrics = calc_metrics(valid_true, valid_predictions, self.class_names)
            if valid_metrics['f1'] > best_val_f1:
                best_val_f1 = valid_metrics['f1']
                best_metrics = valid_metrics

        return best_metrics
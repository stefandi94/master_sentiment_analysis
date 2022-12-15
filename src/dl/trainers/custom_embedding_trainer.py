from consts import MAX_VOCAB_SIZE
from src.dl.trainers import BaseTrainer
from src.dl.utils import get_custom_word_embedding_train_data_loader, get_custom_word_embedding_valid_data_loader, \
    calc_metrics
from src.embeddings.custom_embedding import build_vocab


class CustomWordEmbeddingTrainer(BaseTrainer):
    def train(self, X_train, y_train, X_valid, y_valid, fold):

        vocab = build_vocab(X_train, MAX_VOCAB_SIZE)
        train_loader = get_custom_word_embedding_train_data_loader(X_train, y_train, vocab, self.batch_size)
        valid_loader = get_custom_word_embedding_valid_data_loader(X_valid, y_valid, vocab, self.batch_size)

        best_val_f1, best_loss, best_epoch = 0, 0, 0
        best_model, best_optimizer = None, None
        best_metrics = {}
        valid_true, valid_predictions = [], []

        self.writer.add_text('Model', self.model_name)
        n_model_parameters = sum(p.numel() for p in self.model.parameters())
        self.writer.add_scalar('Model parameters', n_model_parameters)
        print(self.model)
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch: {epoch}")
            self.writer.add_scalar(f"Learning rate", self.scheduler._last_lr[0], epoch)

            true, predictions, train_loss = self.train_loop(self.model, self.optimizer, train_loader)
            self.writer.add_scalar(f"Train loss:", train_loss, epoch)

            valid_true, valid_predictions, valid_loss = self.eval_loop(self.model, valid_loader)
            self.writer.add_scalar(f"Valid loss:", valid_loss, epoch)
            self.scheduler.step()

            train_metrics = calc_metrics(true, predictions)
            valid_metrics = calc_metrics(valid_true, valid_predictions)

            for metric in valid_metrics:
                self.writer.add_scalar(f"Valid metric: {metric}", valid_metrics[metric], epoch)

            for metric in train_metrics:
                self.writer.add_scalar(f"Train metric {metric}", train_metrics[metric], epoch)

            if valid_metrics['f1_macro'] > best_val_f1:
                best_epoch = epoch
                best_val_f1 = valid_metrics['f1_macro']
                best_metrics = valid_metrics
                best_model = self.model
                best_optimizer = self.optimizer
                best_loss = valid_loss
                best_valid_prediction = valid_predictions

        self.save(best_epoch, best_model, best_optimizer, best_loss, best_val_f1, fold)
        return best_metrics, valid_true, best_valid_prediction

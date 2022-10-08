from src.dl.trainers import BaseTrainer
from src.dl.utils import calc_metrics, get_bert_embedding_data_loader


class BertEmbeddingTrainer(BaseTrainer):
    def train(self, X_train, y_train, X_valid, y_valid, tokenizer):
        train_loader = get_bert_embedding_data_loader(X_train, y_train, tokenizer, self.batch_size)
        valid_loader = get_bert_embedding_data_loader(X_valid, y_valid, tokenizer, self.batch_size)

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
import abc
import os
from abc import ABC

import seaborn as sn
import torch
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter

from src.dl.utils import get_model


class BaseTrainer(ABC):

    def __init__(self, model_name, lr, epochs, batch_size, device, classes, optimizer_name, output_dir, log_dir, **model_kwargs):
        self.vocab = None
        self.model_name = model_name
        self.model = get_model(model_name, **model_kwargs).to(device)
        self.model_kwargs = model_kwargs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = self.init_optimizer(optimizer_name)
        self.scheduler = self.init_scheduler()
        self.loss = self.init_loss()
        self.device = device
        self.epochs = epochs
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.writer = self.init_tensorboard_writer()
        self.class_names = classes
        self.n_classes = len(classes)
        self.best_model = None

    @staticmethod
    def init_loss():
        return nn.CrossEntropyLoss()

    def init_scheduler(self):
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)

    def log_params(self):
        self.writer.add_text("Model name", self.model_name)
        for model_param_key, model_param_value in self.model_kwargs:
            self.writer.add_text(model_param_key, model_param_value)

    def init_optimizer(self, optimizer_name):
        return optim.RMSprop(self.model.parameters(), lr=self.lr)

    def init_tensorboard_writer(self):
        os.makedirs(self.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=self.log_dir)
        return writer

    def plot_conf_matrix(self, cm, cm_name, epoch):
        fig = plt.figure(figsize=(10, 7))
        sn.heatmap(cm, annot=True, fmt='.2%', cmap='Blues')
        self.writer.add_figure(f'{cm_name}', fig, global_step=epoch)
        plt.clf()

    def save(self, epoch, model, optimizer, loss, f1, fold):
        print(f'Saving model...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model,
            'optimizer_state_dict': optimizer,
            'loss': loss,
            'f1': f1,
        }, os.path.join(self.output_dir, f"fold_{fold}_val_f1_{f1:.4f}_model.pth"))

    def train_loop(self, model, optimizer, train_loader):
        model.train()

        num_batches = len(train_loader)
        size = len(train_loader.dataset)

        true, predictions = [], []
        train_loss, correct = 0, 0

        log_interval = num_batches // 4

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x = x_batch.type(torch.LongTensor).to(self.device)
            y = y_batch.type(torch.LongTensor).to(self.device)

            y_pred = model(x)

            loss = self.loss(y_pred, y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            pred = y_pred.argmax(dim=1)
            y_true = y.cpu().numpy()

            train_loss += self.loss(y_pred, y).item()
            correct += (pred == y).type(torch.float).sum().item()

            true.extend(y_true)
            predictions.extend(pred.cpu().numpy())

            if batch_idx > 0 and batch_idx % log_interval == 0:
                loss, current = loss.item(), batch_idx * len(x_batch)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        train_loss /= num_batches
        correct /= size
        print(f"Train Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
        return true, predictions, train_loss

    def eval_loop(self, model, eval_loader):
        size = len(eval_loader.dataset)
        num_batches = len(eval_loader)

        predictions = []
        true = []
        test_loss, correct = 0, 0

        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(eval_loader):
                x = x_batch.type(torch.LongTensor).to(self.device)
                y = y_batch.type(torch.LongTensor).to(self.device)

                y_pred = model(x)

                loss = self.loss(y_pred, y).item()
                test_loss += loss

                pred = y_pred.argmax(dim=1)
                y_true = y.cpu().numpy()

                true.extend(y_true)
                predictions.extend(pred.cpu().numpy())
                correct += (pred == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return true, predictions, test_loss

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    # def log_conf_matrix(self, true, preds, mode, epoch):
    #     conf_mat = confusion_matrix(true, preds, labels=list(range(self.n_classes)))
    #     df_cm = pd.DataFrame(conf_mat, index=[label for label in self.class_names],
    #                          columns=[label for label in self.class_names])
    #
    #     fig = plt.figure(figsize=(10, 7))
    #     sn.heatmap(df_cm / np.sum(df_cm), annot=True, fmt='.2%', cmap='Blues')
    #     self.writer.add_figure(f'normalized_{mode}_confusion_matrix', fig, global_step=epoch)
    #     plt.clf()
    #
    #     fig = plt.figure(figsize=(10, 7))
    #     sn.heatmap(df_cm, annot=True, cmap='Blues')
    #     self.writer.add_figure(f'{mode}_relative_confusion_matrix', fig, global_step=epoch)
    #     plt.clf()

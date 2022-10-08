import ast
import itertools
import os

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from transformers import BertTokenizer

from consts import DATASET_PATHS, DATASET_LABEL_TO_INDEX, CLASSIFICATION_MODELS_DIR
from src.dl.trainers.bert_embedding_trainer import BertEmbeddingTrainer
from src.preprocess.data_loading import get_data
from src.utils.utils import save_json
from testing.test_input_arguments import test_dataset_names, test_column_names

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--column_names", type=str, help="Names of the text columns delimited by coma (,)")
    parser.add_argument("--dataset_names", type=str, help="Names of the text columns delimited by coma (,)")
    parser.add_argument("--device", type=str, help="cpu or cuda", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--k_fold", type=int, default=10)

    arguments = parser.parse_args()

    TEXT_COLUMNS = arguments.column_names.split(",")
    DATASET_NAMES = arguments.dataset_names.split(",")
    device = arguments.device
    batch_size = arguments.batch_size
    epochs = arguments.epochs
    lr = arguments.lr
    hidden_dim = arguments.hidden_dim
    k_fold = arguments.k_fold
    log_interval = arguments.log_interval

    test_column_names(TEXT_COLUMNS)
    test_dataset_names(DATASET_NAMES)

    DATASET_PATHS = dict((name, DATASET_PATHS[name]) for name in DATASET_NAMES)
    grids = list(itertools.product(TEXT_COLUMNS, DATASET_PATHS))

    embedding_name = "bert_embeddings"
    global labels
    for grid in tqdm(grids, desc="Grid loop"):
        column_name, dataset_name = grid
        label_mapping = DATASET_LABEL_TO_INDEX[dataset_name]

        output_dir = os.path.join(CLASSIFICATION_MODELS_DIR, dataset_name, embedding_name, column_name)
        os.makedirs(output_dir, exist_ok=True)
        output_model_path = os.path.join(output_dir, "best_estimator.pickle")
        output_grid_search_results_path = os.path.join(output_dir, "grid_search.json")

        if os.path.exists(output_model_path) and os.path.exists(output_grid_search_results_path):
            print(f'Grid search results and model already trained. Continuing!')
            continue

        X, y = get_data(DATASET_PATHS[dataset_name], column_name, label_mapping)

        X = X.reset_index(drop=True)
        splits = KFold(n_splits=k_fold, shuffle=True, random_state=42)
        labels = list(label_mapping.values())

        model_kwargs = {"hidden_dim": hidden_dim, "output_dim": len(labels)}

        best_states = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        for fold, (train_idx, val_idx) in enumerate(tqdm(splits.split(np.arange(len(X))), desc="KFold loop")):
            X_train, X_valid = X[train_idx], X[val_idx]
            y_train, y_valid = y[train_idx], y[val_idx]

            X_train = list(X_train.values)
            X_valid = list(X_valid.values)

            trainer = BertEmbeddingTrainer(
                "bert_embeddings", lr, epochs, batch_size, device, output_dir, output_dir, labels, **model_kwargs
            )
            best_state = trainer.train(X_train, y_train, X_valid, y_valid, tokenizer)
            best_states.append(best_state)

        d = {}
        for k in best_state.keys():
            if "cm" in k:
                continue
            d[k] = np.mean([d[k] for d in best_states])

        d['lr'] = lr
        d.update(**model_kwargs)

        save_json(d, output_grid_search_results_path)

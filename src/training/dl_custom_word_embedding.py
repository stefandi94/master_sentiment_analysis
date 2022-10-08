import itertools
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from consts import DATASET_PATHS, DATASET_LABEL_TO_INDEX, CLASSIFICATION_MODELS_DIR, MAX_VOCAB_SIZE, RESULTS_DIR, \
    LOG_DIR
from src.dl.trainers.custom_embedding_trainer import CustomWordEmbeddingTrainer
from src.preprocess.data_loading import get_data
from src.utils.fit_gridsearch import parse_results
from src.utils.utils import save_json, get_time
from testing.test_input_arguments import test_tokenized_column_names, test_dataset_names

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--column_names", type=str, help="Names of the text columns delimited by coma ,")
    parser.add_argument("--dataset_names", type=str, help="Names of the text columns delimited by coma ,")
    parser.add_argument("--device", type=str, help="cpu or cuda", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--k_fold", type=int, default=10)

    arguments = parser.parse_args()

    TEXT_COLUMNS = arguments.column_names.split(",")
    DATASET_NAMES = arguments.dataset_names.split(",")
    device = arguments.device
    batch_size = arguments.batch_size
    epochs = arguments.epochs
    lr = arguments.lr
    embedding_dim = arguments.embedding_dim
    hidden_dim = arguments.hidden_dim
    k_fold = arguments.k_fold

    test_tokenized_column_names(TEXT_COLUMNS)
    test_dataset_names(DATASET_NAMES)

    DATASET_PATHS = dict((name, DATASET_PATHS[name]) for name in DATASET_NAMES)
    grids = list(itertools.product(TEXT_COLUMNS, DATASET_PATHS))

    model_name = "lstm"
    embedding_name = "custom_word_embedding"

    output_results_path = os.path.join(RESULTS_DIR, "results.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        results = pd.read_csv(output_results_path)
    except FileNotFoundError:
        results = pd.DataFrame()

    for grid in tqdm(grids, desc="Grid loop"):
        print(f'Current grid: {grid}')
        column_name, dataset_name = grid

        log_dir = os.path.join(LOG_DIR, get_time())
        label_mapping = DATASET_LABEL_TO_INDEX[dataset_name]

        X, y = get_data(DATASET_PATHS[dataset_name], column_name, label_mapping)

        X = X.reset_index(drop=True)
        splits = KFold(n_splits=k_fold, shuffle=True, random_state=42)
        labels = list(label_mapping.values())

        model_kwargs = {"input_dim": MAX_VOCAB_SIZE + 2, "embedding_dim": embedding_dim, "hidden_dim": hidden_dim, "output_dim": len(labels)}

        best_states = []
        for fold, (train_idx, val_idx) in enumerate(tqdm(splits.split(np.arange(len(X))), desc="KFold loop")):
            X_train, X_valid = X[train_idx], X[val_idx]
            y_train, y_valid = y[train_idx], y[val_idx]

            trainer = CustomWordEmbeddingTrainer(
                "custom_embeddings", lr, epochs, batch_size, device, labels, log_dir, **model_kwargs
            )
            best_state = trainer.train(X_train, y_train, X_valid, y_valid)
            best_states.append(best_state)

        d = {}
        for k in best_state.keys():
            if "cm" in k:
                continue
            d[k] = np.mean([d[k] for d in best_states])

        # model_kwargs["lr"] = lr
        # model_kwargs["epoch"] = epochs
        # model_kwargs["batch_size"] = batch_size

        parsed_results = parse_results(best_states, model_kwargs, lr, epochs, batch_size)

        # results = pd.concat([results, pd.DataFrame(parsed_results, index=[results.shape[1]])])

        # results.to_csv(output_results_path, index=False)

        # save_json(d, output_grid_search_results_path)

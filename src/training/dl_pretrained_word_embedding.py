import itertools
import os

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from consts import DATASET_PATHS, DATASET_LABEL_TO_INDEX, CLASSIFICATION_MODELS_DIR
from src.dl.trainers.word_embedding_trainer import PretrainedWordEmbeddingTrainer
from src.embeddings.pretrained_embedding import load_gensim_embeddings
from src.preprocess.data_loading import get_data
from src.utils.utils import save_json
from testing.test_input_arguments import test_tokenized_column_names, test_dataset_names

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--column_names", type=str, help="Names of the text columns delimited by coma (,)")
    parser.add_argument("--dataset_names", type=str, help="Names of the text columns delimited by coma (,)")
    parser.add_argument("--embedding_name", type=str, help="Names of pretrained embedding")
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
    embedding_name = arguments.embedding_name
    hidden_dim = arguments.hidden_dim
    k_fold = arguments.k_fold
    log_interval = arguments.log_interval

    test_tokenized_column_names(TEXT_COLUMNS)
    test_dataset_names(DATASET_NAMES)

    DATASET_PATHS = dict((name, DATASET_PATHS[name]) for name in DATASET_NAMES)
    grids = list(itertools.product(TEXT_COLUMNS, DATASET_PATHS))

    model_name = "word_embeddings"
    embedding_dim = 300
    global labels
    for grid in tqdm(grids, desc="Grid loop"):
        print(f'Current grid: {grid}')
        column_name, dataset_name = grid
        if not results.empty:
            if [model_name, embedding_name, column_name, dataset_name] in results[['model_name', 'embedding_name', 'column_name', 'dataset_name']].values.tolist():
                print(f'Continuing, parameters already trained!')
                continue
        label_mapping = DATASET_LABEL_TO_INDEX[dataset_name]

        output_dir = os.path.join(CLASSIFICATION_MODELS_DIR, dataset_name, model_name, embedding_name)
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

        embedding = load_gensim_embeddings(embedding_name)
        model_kwargs = {"embedding": embedding, "hidden_dim": hidden_dim, "output_dim": len(labels)}

        best_states = []
        for fold, (train_idx, val_idx) in enumerate(tqdm(splits.split(np.arange(len(X))), desc="KFold loop")):
            X_train, X_valid = X[train_idx], X[val_idx]
            y_train, y_valid = y[train_idx], y[val_idx]

            trainer = PretrainedWordEmbeddingTrainer(model_name, lr, epochs, batch_size, device, output_dir, output_dir, labels, **model_kwargs)
            best_state = trainer.train(X_train, y_train, X_valid, y_valid, embedding)
            best_states.append(best_state)

        d = {}
        for k in best_state.keys():
            if "cm" in k:
                continue
            d[k] = np.mean([d[k] for d in best_states])

        d['lr'] = lr
        d.update(**model_kwargs)

        save_json(d, output_grid_search_results_path)

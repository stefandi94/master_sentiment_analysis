import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from consts import DATASET_PATHS, DATASET_LABEL_TO_INDEX, RESULTS_DIR, LOG_DIR, CLASSIFICATION_MODELS_DIR, CV
from src.dl.trainers.transformers_embedding_trainer import TransformersEmbeddingTrainer
from src.dl.utils import get_transformer_embedding
from src.preprocess.data_loading import get_data
from src.utils.fit_gridsearch import parse_results
from src.utils.utils import get_time
from src.utils.visualization import plot_conf_matrix, plot_clf_report
from testing.test_input_arguments import test_dataset_names, test_column_names

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--column_names", type=str, help="Names of the text columns delimited by coma ,",
                        default="clean_text")
    parser.add_argument("--dataset_names", type=str, help="Names of the text columns delimited by coma (,)")
    parser.add_argument("--device", type=str, help="cpu or cuda", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--k_fold", type=int, default=CV)
    parser.add_argument("--transformer_name", type=str, default="bert-base-uncased")
    parser.add_argument("--tokenizer_name", type=str, default='bert-base-uncased')
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=False)
    parser.add_argument("--optimizer", type=str, default="rmsprop")
    parser.add_argument("--last_feature", type=bool, default=True)
    parser.add_argument("--avg_pool", type=bool, default=True)
    parser.add_argument("--max_pool", type=bool, default=True)

    arguments = parser.parse_args()

    TEXT_COLUMNS = arguments.column_names.split(",")
    DATASET_NAMES = arguments.dataset_names.split(",")
    device = arguments.device
    batch_size = arguments.batch_size
    epochs = arguments.epochs
    lr = arguments.lr
    hidden_dim = arguments.hidden_dim
    k_fold = arguments.k_fold
    transformer_name = arguments.transformer_name
    tokenizer_name = arguments.tokenizer_name
    dropout = arguments.dropout
    bidirectional = arguments.bidirectional
    optimizer = arguments.optimizer
    last_feature = arguments.last_feature
    avg_pool = arguments.avg_pool
    max_pool = arguments.max_pool

    test_column_names(TEXT_COLUMNS)
    test_dataset_names(DATASET_NAMES)

    DATASET_PATHS = dict((name, DATASET_PATHS[name]) for name in DATASET_NAMES)
    grids = list(itertools.product(TEXT_COLUMNS, DATASET_PATHS))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    transformer_model = AutoModel.from_pretrained(transformer_name)
    embeddings = get_transformer_embedding(transformer_model)
    embedding_dim = transformer_model.embeddings.word_embeddings.weight.shape[-1]
    embedding_name = transformer_name
    model_name = "transformer_embeddings"

    output_results_path = os.path.join(RESULTS_DIR, "results.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        results = pd.read_csv(output_results_path)
    except FileNotFoundError:
        results = pd.DataFrame()

    for grid in tqdm(grids, desc="Grid loop"):
        column_name, dataset_name = grid
        time = get_time()
        print(f'Current grid: {grid} at {time}')

        base_output_dir = os.path.join(CLASSIFICATION_MODELS_DIR, dataset_name, column_name, model_name, time)
        base_log_dir = os.path.join(LOG_DIR, dataset_name, column_name, model_name, time)

        label_mapping = DATASET_LABEL_TO_INDEX[dataset_name]
        X, y = get_data(DATASET_PATHS[dataset_name], column_name, label_mapping)

        X = X.reset_index(drop=True)
        splits = KFold(n_splits=k_fold, shuffle=True, random_state=42)
        labels = list(label_mapping.values())

        best_states = []
        true_labels = []
        prediction_labels = []
        for fold, (train_idx, val_idx) in enumerate(tqdm(splits.split(np.arange(len(X))), desc="KFold loop")):
            model_kwargs = {
                "hidden_dim": hidden_dim, "output_dim": len(labels),
                "bidirectional": bidirectional, "dropout": dropout, "embeddings": embeddings,
                "last_feature": last_feature, "avg_pool": avg_pool, "max_pool": max_pool
            }
            print(f'Fold number: {fold}')
            log_dir = os.path.join(base_log_dir, str(fold))
            output_dir = os.path.join(base_output_dir, str(fold))
            os.makedirs(output_dir, exist_ok=True)

            X_train, X_valid = X[train_idx], X[val_idx]
            y_train, y_valid = y[train_idx], y[val_idx]

            X_train = list(X_train.values)
            X_valid = list(X_valid.values)

            trainer = TransformersEmbeddingTrainer(
                model_name, lr, epochs, batch_size, device, labels, optimizer, output_dir, log_dir, **model_kwargs
            )
            print(trainer.model)

            trainable_parameters = sum(p.numel() for p in trainer.model.parameters())
            print(f'Number of trainable parameters: {trainable_parameters}')

            non_trainable_parameters = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            print(f'Number of non-trainable parameters: {non_trainable_parameters}')

            best_state, true, predictions = trainer.train(X_train, y_train, X_valid, y_valid, tokenizer, fold)

            true_labels.extend(true)
            prediction_labels.extend(predictions)
            best_states.append(best_state)

        conf_matrix = confusion_matrix(true_labels, prediction_labels, labels=range(len(label_mapping)))
        plot_conf_matrix(conf_matrix, base_output_dir, list(label_mapping.keys()))

        clf_report = classification_report(true_labels, prediction_labels, target_names=list(label_mapping.keys()), labels=range(len(list(label_mapping.keys()))), output_dict=True)
        plot_clf_report(clf_report, os.path.join(base_output_dir, "classification_report"))

        d = {}
        for k in best_state.keys():
            if "cm" in k:
                continue
            d[k] = np.mean([d[k] for d in best_states])

        del model_kwargs['embeddings']
        model_kwargs['embedding_dim'] = embedding_dim
        model_kwargs['transformer_name'] = transformer_name
        model_kwargs['tokenizer_name'] = tokenizer_name
        model_kwargs['optimizer_name'] = optimizer
        model_kwargs['trainable_parameters'] = trainable_parameters
        model_kwargs['non_trainable'] = non_trainable_parameters

        parsed_results = parse_results(d, model_name, column_name, dataset_name, embedding_name, model_kwargs, lr,
                                       epochs, batch_size)
        parsed_results["classification_report"] = str(clf_report)
        parsed_results["confusion_matrix"] = str(conf_matrix)

        results = pd.concat([results, pd.DataFrame(parsed_results, index=[results.shape[1]])])
        results.to_csv(output_results_path, index=False)

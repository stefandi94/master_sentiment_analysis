import itertools
import os
from pprint import pprint

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm

from consts import DATASET_LABEL_TO_INDEX, RESULTS_DIR, CLASSIFICATION_MODELS_DIR, CV
from consts import DATASET_PATHS
from src.grid_parameters.tf_idf import get_model_grid
from src.preprocess.data_loading import get_data
from src.utils.fit_gridsearch import fit_grid, parse_grid_search_results
from src.utils.visualization import plot_conf_matrix, plot_clf_report
from testing.test_input_arguments import test_column_names, test_model_names, test_dataset_names

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", type=str, help="Names of the classification models delimited by ,")
    parser.add_argument("--column_names", type=str, help="Names of the text columns delimited by ,",
                        default="clean_text")
    parser.add_argument("--dataset_names", type=str, help="Names of the text columns delimited by ,")

    arguments = parser.parse_args()

    MODEL_NAMES = arguments.model_names.split(",")
    TEXT_COLUMNS = arguments.column_names.split(",")
    DATASET_NAMES = arguments.dataset_names.split(",")

    test_column_names(TEXT_COLUMNS)
    test_dataset_names(DATASET_NAMES)
    test_model_names(MODEL_NAMES)

    DATASET_PATHS = dict((name, DATASET_PATHS[name]) for name in DATASET_NAMES)
    embedding_name = "tfidf"
    grids = list(itertools.product(MODEL_NAMES, TEXT_COLUMNS, DATASET_PATHS))

    output_results_path = os.path.join(RESULTS_DIR, "results.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        results = pd.read_csv(output_results_path)
    except FileNotFoundError:
        results = pd.DataFrame()

    for grid in tqdm(grids, desc="Grid loop"):
        pprint(f'Current grid: {grid}')

        model_name, column_name, dataset_name = grid
        if not results.empty:
            if [model_name, embedding_name, column_name, dataset_name] in results[['model_name', 'embedding_name', 'column_name', 'dataset_name']].values.tolist():
                print(f'Continuing, parameters already trained!')
                continue

        label_mapping = DATASET_LABEL_TO_INDEX[dataset_name]
        output_dir = os.path.join(CLASSIFICATION_MODELS_DIR, dataset_name, column_name, model_name)
        os.makedirs(output_dir, exist_ok=True)
        X, y = get_data(DATASET_PATHS[dataset_name], column_name, label_mapping)

        parameter_grid = get_model_grid(model_name)
        parameters = parameter_grid['parameters']
        pipe = parameter_grid['pipe']

        grid_search = fit_grid(X, y, parameter_grid['pipe'], parameter_grid['parameters'])

        y_pred = cross_val_predict(grid_search.best_estimator_, X, y, cv=CV)

        conf_matrix = confusion_matrix(y, y_pred, labels=range(len(label_mapping)))
        plot_conf_matrix(conf_matrix, output_dir, list(label_mapping.keys()))

        clf_report = classification_report(y, y_pred, target_names=list(label_mapping.keys()), labels=range(len(list(label_mapping.keys()))), output_dict=True)
        plot_clf_report(clf_report, os.path.join(output_dir, "classification_report"))

        parsed_results = parse_grid_search_results(grid_search, model_name, embedding_name, column_name, dataset_name)
        parsed_results["classification_report"] = str(clf_report)
        parsed_results["confusion_matrix"] = str(conf_matrix)

        results = pd.concat([results, pd.DataFrame(parsed_results, index=[results.shape[1]])])
        results.to_csv(output_results_path, index=False)

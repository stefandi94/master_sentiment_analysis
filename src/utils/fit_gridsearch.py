from sklearn.model_selection import GridSearchCV

scoring = ['accuracy', 'recall_micro', 'precision_micro', 'f1_micro']


def fit_grid(X, y, pipe, parameters):
    gs = GridSearchCV(pipe, parameters, n_jobs=8, verbose=0, cv=10, scoring=scoring, refit='f1_micro')
    gs.fit(X, y)
    return gs


def parse_results(best_states, model_name, column_name, dataset_name, embedding_name, model_kwargs, lr, epochs,
                  batch_size):
    best_parameters = model_kwargs.copy()
    best_parameters["lr"] = lr
    best_parameters["epoch"] = epochs
    best_parameters["batch_size"] = batch_size
    best_states['best_params'] = str(best_parameters)
    best_states['embedding_name'] = embedding_name
    best_states['model_name'] = model_name
    best_states['column_name'] = column_name
    best_states['dataset_name'] = dataset_name
    return best_states


def parse_grid_search_results(grid_search, model_name, embedding_name, column_name, dataset_name):
    best_index = grid_search.best_index_
    best_params = grid_search.best_params_

    best_accuracy = grid_search.cv_results_['mean_test_accuracy'][best_index]
    best_recall = grid_search.cv_results_['mean_test_recall_micro'][best_index]
    best_precision = grid_search.cv_results_['mean_test_precision_micro'][best_index]
    best_f1_macro = grid_search.cv_results_['mean_test_f1_micro'][best_index]

    grid_search_results = {
        "accuracy": round(best_accuracy, 4), "recall_micro": round(best_recall, 4),
        "precision_micro": round(best_precision, 4), "f1_micro": round(best_f1_macro, 4),
        'model_name': model_name, 'embedding_name': embedding_name, 'column_name': column_name,
        'dataset_name': dataset_name, "best_params": str(best_params)
    }
    return grid_search_results
